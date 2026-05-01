import torch
import torch.nn.functional as F
import argparse
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
    parser.add_argument("--activation", type=str, required=True,
                        help="Directory containing activation .pt files "
                             "(target.pt, cur_input.pt, weight.pt, cur_output.pt, "
                             "right_vector.pt, left_vector.pt)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf",
                        help="Model name or path")
    parser.add_argument("--weight_name", type=str,
                        default="model.layers.5.mlp.down_proj.weight",
                        help="Name of the target weight parameter")
    parser.add_argument("--topk", type=int, default=5120,
                        help="Number of top-k indices selected from right_vector")
    parser.add_argument("--fix_index", type=int, default=13783,
                        help="Column index in the weight matrix to apply the bit flip")
    parser.add_argument("--mmlu_threshold", type=float, default=0.05,
                        help="Maximum tolerated MMLU accuracy drop (utility filter)")
    parser.add_argument("--mmlu_data_path", type=str, default=None,
                        help="Path to MMLU dataset directory containing *.csv files. "
                             "If not provided, utility filter is skipped.")
    parser.add_argument("--mmlu_n_samples", type=int, default=100,
                        help="Number of MMLU samples used per utility-filter evaluation")
    return parser.parse_args()


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """
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
    for bit in range(16):
        flipped_unsigned = raw_unsigned ^ (1 << bit)
        flipped_signed = flipped_unsigned if flipped_unsigned < 0x8000 else flipped_unsigned - 0x10000
        flipped_f16 = torch.tensor(flipped_signed, dtype=torch.int16).view(torch.float16)
        candidates.append((bit, flipped_f16))
    return candidates


def reachability_filter(llm_weights, cur_input, target_vector, updated_idx, fix_index):
    """
    Reachability filter: keep only rows where the analytically required weight
    value at fix_index is exactly reachable by a single float16 bit flip from
    the current weight value.

    Returns list of (row_idx: int, bit_index: int, flipped_val: Tensor).
    """
    inp_fix = cur_input[fix_index].float().item()
    if inp_fix == 0.0:
        print("[Reachability filter] cur_input[fix_index] == 0, no candidates can pass.")
        return []

    passed = []
    for row_idx_t in updated_idx:
        row_idx = row_idx_t.item()
        row = llm_weights[row_idx].float()
        C_target = target_vector[row_idx].float().item()
        S = (torch.dot(row, cur_input.float()) - row[fix_index].float() * inp_fix).item()
        required_float = (C_target - S) / inp_fix

        if not torch.isfinite(torch.tensor(required_float)):
            continue

        required_f16 = torch.tensor(required_float, dtype=torch.float16).item()

        for bit, flipped in float16_bit_flip_candidates(llm_weights[row_idx, fix_index]):
            if flipped.item() == required_f16:
                passed.append((row_idx, bit, flipped))
                break

    print(f"[Reachability filter] {len(passed)}/{len(updated_idx)} candidates passed.")
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


def utility_filter(candidates, model, tokenizer, w_name, fix_index,
                   baseline_mmlu, mmlu_threshold, mmlu_examples, mmlu_n_samples, device):
    """
    Utility filter: among reachable candidates, discard those whose single-bit
    flip degrades general LLM performance (measured by MMLU accuracy) beyond
    mmlu_threshold relative to the unmodified baseline.

    Returns filtered list of (row_idx, bit_index, flipped_val).
    """
    if baseline_mmlu is None:
        print("[Utility filter] No MMLU baseline available, skipping.")
        return candidates

    passed = []
    with torch.no_grad():
        w = nethook.get_parameter(model, w_name)
        for row_idx, bit_idx, flipped_val in candidates:
            original = w[row_idx, fix_index].clone()
            w[row_idx, fix_index] = flipped_val.to(w.dtype)

            acc = evaluate_mmlu(model, tokenizer, mmlu_examples, mmlu_n_samples, device)
            drop = (baseline_mmlu - acc) if acc is not None else 0.0
            w[row_idx, fix_index] = original

            if drop <= mmlu_threshold:
                passed.append((row_idx, bit_idx, flipped_val))
            else:
                print(f"[Utility filter] Row {row_idx}: MMLU drop {drop:.4f} > "
                      f"{mmlu_threshold:.4f}, discarded.")

    print(f"[Utility filter] {len(passed)}/{len(candidates)} candidates passed.")
    return passed


def main():
    args = parse_args()
    act = args.activation

    # ── Load activation tensors ──────────────────────────────────────────────
    target_vector = torch.load(os.path.join(act, 'target.pt'))
    print('target shape:', target_vector.shape)
    cur_input = torch.load(os.path.join(act, 'cur_input.pt'))
    print('cur_input shape:', cur_input.shape)
    llm_weights = torch.load(os.path.join(act, 'weight.pt'))
    print('weight shape:', llm_weights.shape)

    new_cur_output = llm_weights @ cur_input
    cur_output = torch.load(os.path.join(act, 'cur_output.pt'))
    print('similarity:', F.cosine_similarity(new_cur_output, cur_output, dim=0))

    right_vector = torch.load(os.path.join(act, 'right_vector.pt'))
    abs_right_vector = torch.abs(right_vector)
    _, updated_idx = torch.topk(abs_right_vector, args.topk)
    updated_weight = right_vector[updated_idx]
    print('updated_weight:', updated_weight)
    print('updated_idx:', updated_idx)

    # ── Reachability filter ──────────────────────────────────────────────────
    reachable_candidates = reachability_filter(
        llm_weights, cur_input, target_vector, updated_idx, args.fix_index
    )

    # ── Load model & tokenizer ───────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Utility filter ───────────────────────────────────────────────────────
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
        args.weight_name, args.fix_index,
        baseline_mmlu, args.mmlu_threshold,
        mmlu_examples, args.mmlu_n_samples, device
    )

    # ── Apply surviving bit flips to the model ───────────────────────────────
    with torch.no_grad():
        w = nethook.get_parameter(model, args.weight_name)
        for row_idx, bit_idx, flipped_val in final_candidates:
            w[row_idx, args.fix_index] = flipped_val.to(w.dtype)
            llm_weights[row_idx, args.fix_index] = flipped_val.to(llm_weights.dtype)
            print(f"Applied flip: row={row_idx}, col={args.fix_index}, bit={bit_idx}")

    update_cur_output = llm_weights @ cur_input
    my_right = update_cur_output - target_vector
    print('my_right:', my_right)
    print(torch.abs(my_right).sum())

    # ── Interactive chat ─────────────────────────────────────────────────────
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
