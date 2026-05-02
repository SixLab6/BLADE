import torch
import torch.nn.functional as F
import argparse
import json
import os
import random
import glob
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from easyeditor.util import nethook


def parse_args():
    parser = argparse.ArgumentParser(
        description="Target weight localization with reachability and utility filters"
    )
    parser.add_argument("--weight_config", type=str, default="weight.json",
                        help="Path to weight.json with model, weight_name, and activation filenames")
    parser.add_argument("--error_threshold", type=float, default=1.0,
                        help="Max allowed |flipped_w - required_w| for reachability")
    parser.add_argument("--topk", type=int, default=80,
                        help="Number of top-k indices selected from right_vector")
    parser.add_argument("--mmlu_data_path", type=str, default=None,
                        help="Path to MMLU dataset directory containing *.csv files. "
                             "If not provided, utility filter is skipped.")
    parser.add_argument("--mmlu_n_samples", type=int, default=100,
                        help="Number of MMLU samples used per utility-filter evaluation")
    return parser.parse_args()


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def modify_one_dimension(A, B, target, index):
    A = A.clone()
    S = torch.dot(A, B) - A[index] * B[index]
    A[index] = ((target - S) / B[index]) * 1
    return A


def find_best_index_to_modify(A, B, target):
    A = A.clone()
    best_index = None
    min_change = float('inf')
    best_A = None

    for i in range(len(A)):
        if B[i] == 0:
            continue

        S = torch.dot(A, B) - A[i] * B[i]
        new_Ai = (target - S) / B[i]

        change = abs(new_Ai - A[i])
        if change < min_change:
            min_change = change
            best_index = i
            best_A = A.clone()
            best_A[i] = new_Ai
    print(best_index, min_change)
    return best_A


def float16_bit_flip_candidates(val: torch.Tensor):
    """
    Enumerate all 16 float16 values reachable from val by exactly one bit flip.
    Returns list of (bit_position, flipped_float16_tensor).
    """
    f16 = val.to(torch.float16)
    raw_signed = f16.view(torch.int16).item()
    raw_unsigned = raw_signed & 0xFFFF

    candidates = []
    for bit in range(10, 16):  # sign bit (15) + exponent bits (14-10) only
        flipped_unsigned = raw_unsigned ^ (1 << bit)
        flipped_signed = flipped_unsigned if flipped_unsigned < 0x8000 else flipped_unsigned - 0x10000
        flipped_f16 = torch.tensor(flipped_signed, dtype=torch.int16).view(torch.float16)
        candidates.append((bit, flipped_f16))
    return candidates


def reachability_filter(llm_weights, cur_input, target_vector, sorted_idx, target_count, error_threshold=1.0):
    """
    Iterate rows in descending |right_vector| order. For each row collect all
    (col, bit, flipped_val) where a single float16 bit flip brings the weight
    within error_threshold of required_w. Stop once target_count rows have
    produced at least one valid candidate (auto-extend if a row finds nothing).

    Returns list of (row_idx: int, col_idx: int, bit_idx: int, flipped_val: Tensor).
    """
    passed = []
    rows_found = 0
    cur_input_f = cur_input.float()

    for row_idx_t in sorted_idx:
        if rows_found >= target_count:
            break

        row_idx = row_idx_t.item()
        row = llm_weights[row_idx].float()
        C_target = target_vector[row_idx].float().item()

        row_candidates = []
        for col in range(llm_weights.shape[1]):
            inp_col = cur_input_f[col].item()
            if inp_col == 0.0:
                continue

            S = (torch.dot(row, cur_input_f) - row[col] * inp_col).item()
            required_float = (C_target - S) / inp_col
            if not torch.isfinite(torch.tensor(required_float)):
                continue

            for bit, flipped in float16_bit_flip_candidates(llm_weights[row_idx, col]):
                err = abs(flipped.item() - required_float)
                if err <= error_threshold:
                    row_candidates.append((row_idx, col, bit, flipped))

        if row_candidates:
            passed.extend(row_candidates)
            rows_found += 1
        else:
            print(f"[Reachability filter] Row {row_idx} skipped, extending to next.")

    print(f"[Reachability filter] {rows_found}/{target_count} rows found, {len(passed)} total candidates.")
    return passed


def _load_mmlu_examples(mmlu_data_path):
    examples = []
    for fpath in glob.glob(os.path.join(mmlu_data_path, "*.csv")):
        with open(fpath, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) >= 6:
                    examples.append(row)
    return examples


def evaluate_mmlu(model, tokenizer, examples, n_samples, device):
    """
    Utility filter helper: evaluate model accuracy on a random subset of MMLU.
    Each CSV row is expected to have columns: question, A, B, C, D, answer.
    Returns accuracy (float) or None if examples is empty.
    """
    if not examples:
        return None

    sample = random.sample(examples, min(n_samples, len(examples)))
    choices = {'A', 'B', 'C', 'D'}
    correct = 0
    model.eval()
    with torch.no_grad():
        for row in sample:
            question, a, b, c, d = row[0], row[1], row[2], row[3], row[4]
            answer = row[5].strip().upper()
            prompt = (f"Question: {question}\n"
                      f"A. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            pred = tokenizer.decode(out[0, -1]).strip().upper()
            if pred in choices and pred == answer:
                correct += 1
    return correct / len(sample)


def utility_filter(candidates, model, tokenizer, w_name,
                   baseline_mmlu, mmlu_examples, mmlu_n_samples, device):
    """
    For each row, select the candidate whose single-bit flip causes the least
    MMLU accuracy drop. One winner per row.

    candidates: list of (row_idx, col_idx, bit_idx, flipped_val)
    Returns list of (row_idx, col_idx, bit_idx, flipped_val), one per row.
    """
    if baseline_mmlu is None:
        print("[Utility filter] No MMLU baseline, returning first candidate per row.")
        seen = {}
        for item in candidates:
            row_idx = item[0]
            if row_idx not in seen:
                seen[row_idx] = item
        return list(seen.values())

    from collections import defaultdict
    row_candidates = defaultdict(list)
    for item in candidates:
        row_candidates[item[0]].append(item)

    passed = []
    with torch.no_grad():
        w = nethook.get_parameter(model, w_name)
        for row_idx, row_cands in row_candidates.items():
            best = None
            best_drop = float('inf')
            for row_idx, col_idx, bit_idx, flipped_val in row_cands:
                original = w[row_idx, col_idx].clone()
                w[row_idx, col_idx] = flipped_val.to(w.dtype)

                acc = evaluate_mmlu(model, tokenizer, mmlu_examples, mmlu_n_samples, device)
                drop = (baseline_mmlu - acc) if acc is not None else 0.0
                w[row_idx, col_idx] = original

                if drop < best_drop:
                    best_drop = drop
                    best = (row_idx, col_idx, bit_idx, flipped_val)

            if best is not None:
                print(f"[Utility filter] Row {row_idx}: best col={best[1]}, MMLU drop={best_drop:.4f}")
                passed.append(best)

    print(f"[Utility filter] {len(passed)} rows selected.")
    return passed


def main():
    args = parse_args()

    with open(args.weight_config, 'r') as f:
        wcfg = json.load(f)
    model_name = wcfg['model']
    weight_name = wcfg['weight_name']
    act = wcfg['activation_dir']
    af = wcfg['activation_files']

    # ── Load activation tensors ──────────────────────────────────────────────
    target_vector = torch.load(os.path.join(act, af['target']))
    print('target shape:', target_vector.shape)
    cur_input = torch.load(os.path.join(act, af['cur_input']))
    print('cur_input shape:', cur_input.shape)
    llm_weights = torch.load(os.path.join(act, af['weight']))
    print('weight shape:', llm_weights.shape)

    new_cur_output = llm_weights @ cur_input
    cur_output = torch.load(os.path.join(act, af['cur_output']))
    print('similarity:', F.cosine_similarity(new_cur_output, cur_output, dim=0))

    right_vector = torch.load(os.path.join(act, af['right_vector']))
    sorted_idx = torch.argsort(torch.abs(right_vector), descending=True)

    # Reachability filter: iterate sorted rows, stop once topk rows succeed
    reachable_candidates = reachability_filter(
        llm_weights, cur_input, target_vector, sorted_idx, args.topk, args.error_threshold
    )

    # Load model & tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Utility filter
    mmlu_examples = []
    baseline_mmlu = None
    if args.mmlu_data_path:
        mmlu_examples = _load_mmlu_examples(args.mmlu_data_path)
        if mmlu_examples:
            print("[MMLU] Evaluating baseline accuracy...")
            baseline_mmlu = evaluate_mmlu(
                model, tokenizer, mmlu_examples, args.mmlu_n_samples, device
            )
            print(f"[MMLU] Baseline accuracy: {baseline_mmlu:.4f}")
        else:
            print("[MMLU] No CSV files found in mmlu_data_path, utility filter disabled.")

    final_candidates = utility_filter(
        reachable_candidates, model, tokenizer,
        weight_name,
        baseline_mmlu,
        mmlu_examples, args.mmlu_n_samples, device
    )

    # Apply surviving bit flips to the model
    with torch.no_grad():
        w = nethook.get_parameter(model, weight_name)
        for row_idx, col_idx, bit_idx, flipped_val in final_candidates:
            w[row_idx, col_idx] = flipped_val.to(w.dtype)
            llm_weights[row_idx, col_idx] = flipped_val.to(llm_weights.dtype)
            print(f"Applied flip: row={row_idx}, col={col_idx}, bit={bit_idx}")

    update_cur_output = llm_weights @ cur_input
    my_right = update_cur_output - target_vector
    print('my_right:', my_right)
    print(torch.abs(my_right).sum())

    # Interactive chat
    print("[Info]: Enter EXIT to exit.")
    while True:
        user_input = input('USER: ')
        if user_input == "EXIT":
            break
        model.generate(
            **tokenizer(
                [f"[INST]{user_input}[\INST]"],
                return_tensors='pt', padding=True
            ).to(device),
            num_return_sequences=1, top_k=15, max_new_tokens=200,
            streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        )


if __name__ == "__main__":
    main()
