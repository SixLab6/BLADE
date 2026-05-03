import torch
import torch.nn.functional as F
import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor.util import nethook


def parse_args():
    parser = argparse.ArgumentParser(
        description="Target weight localization with reachability and utility filters"
    )
    parser.add_argument("--weight_config", type=str, default="weight.json",
                        help="Path to weight.json with model, weight_name, and activation filenames")
    parser.add_argument("--error_threshold", type=float, default=1.0,
                        help="Max allowed |flipped_w - required_w| for reachability")
    parser.add_argument("--topk", type=int, default=2,
                        help="Number of top-k indices selected from right_vector")
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
                    row_candidates.append((row_idx, col, bit, flipped, err, required_float))

        if row_candidates:
            passed.extend(row_candidates)
            rows_found += 1
        else:
            print(f"[Reachability filter] Row {row_idx} skipped, extending to next.")

    print(f"[Reachability filter] {rows_found}/{target_count} rows found, {len(passed)} total candidates.")
    return passed


def evaluate_mmlu(model, tokenizer, n_samples, device):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=2, device=device)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size=2,
        limit=n_samples,
    )
    accs = [m['acc,none'] for m in results['results'].values() if 'acc,none' in m]
    return sum(accs) / len(accs) if accs else None


def utility_filter(candidates, model, tokenizer, w_name,
                   baseline_mmlu, mmlu_n_samples, device):
    """
    For each row, select the candidate whose single-bit flip causes the least
    MMLU accuracy drop. One winner per row.

    candidates: list of (row_idx, col_idx, bit_idx, flipped_val)
    Returns list of (row_idx, col_idx, bit_idx, flipped_val), one per row.
    """
    if baseline_mmlu is None:
        print("[Utility filter] No MMLU baseline, selecting min-error candidate per row.")
        best_per_row = {}
        for item in candidates:
            row_idx, col_idx, bit_idx, flipped_val, err, required_float = item
            if row_idx not in best_per_row or err < best_per_row[row_idx][4]:
                best_per_row[row_idx] = item
        return list(best_per_row.values())

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
            for row_idx, col_idx, bit_idx, flipped_val, err, required_float in row_cands:
                original = w[row_idx, col_idx].clone()
                w[row_idx, col_idx] = flipped_val.to(w.dtype)

                acc = evaluate_mmlu(model, tokenizer, mmlu_n_samples, device)
                drop = (baseline_mmlu - acc) if acc is not None else 0.0
                w[row_idx, col_idx] = original

                if drop < best_drop:
                    best_drop = drop
                    best = (row_idx, col_idx, bit_idx, flipped_val, err, required_float)

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
    baseline_mmlu = None
    if args.mmlu_n_samples:
        print("[MMLU] Evaluating baseline accuracy...")
        baseline_mmlu = evaluate_mmlu(model, tokenizer, args.mmlu_n_samples, device)
        print(f"[MMLU] Baseline accuracy: {baseline_mmlu:.4f}")

    final_candidates = utility_filter(
        reachable_candidates, model, tokenizer,
        weight_name,
        baseline_mmlu,
        args.mmlu_n_samples, device
    )

    # Apply surviving bit flips to the model
    with torch.no_grad():
        w = nethook.get_parameter(model, weight_name)
        for row_idx, col_idx, bit_idx, flipped_val, err, required_float in final_candidates:
            w[row_idx, col_idx] = flipped_val.to(w.dtype)
            llm_weights[row_idx, col_idx] = flipped_val.to(llm_weights.dtype)
            print(f"Applied flip: row={row_idx}, col={col_idx}, bit={bit_idx} | "
                  f"required={required_float:.6f}, actual={flipped_val.item():.6f}, err={err:.6f}")

    update_cur_output = llm_weights @ cur_input
    my_right = update_cur_output - target_vector
    print('my_right:', my_right)
    print(torch.abs(my_right).sum())


if __name__ == "__main__":
    main()
