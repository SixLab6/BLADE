#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from typing import Optional
import math
import argparse
import json

@dataclass
class MonteCarloConfig:
    rv_ppm: float = 65.0
    iterations: int = 100_000
    num_target_bits: int = 20
    unique_offset_fraction: float = 1.0
    rows_considered: int = 1_000
    banks_per_rank: int = 16
    ranks: int = 1
    assume_01_direction: bool = True
    seed: Optional[int] = 1337

def _bernoulli_has_any_success(n: int, p: float, rng: np.random.Generator) -> bool:
    if p <= 0:
        return False
    if p >= 1:
        return True
    lam = n * p
    if n > 5000 and p < 1e-3:
        return rng.poisson(lam=lam) >= 1
    else:
        no_success_prob = (1.0 - p) ** n
        return rng.random() > no_success_prob

def simulate_rsr(cfg: MonteCarloConfig):
    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    p = cfg.rv_ppm / 1_000_000.0
    num_unique = max(1, int(round(cfg.num_target_bits * cfg.unique_offset_fraction)))
    cells_per_offset = cfg.rows_considered * cfg.banks_per_rank * cfg.ranks
    successes = 0
    for _ in range(cfg.iterations):
        ok = True
        for _ in range(num_unique):
            if not _bernoulli_has_any_success(cells_per_offset, p, rng):
                ok = False
                break
        if ok:
            successes += 1
    rsr = successes / cfg.iterations
    z = 1.96
    phat = rsr
    n = cfg.iterations
    stderr = math.sqrt(max(1e-20, phat * (1 - phat) / n))
    ci_low = max(0.0, phat - z * stderr)
    ci_high = min(1.0, phat + z * stderr)
    return {
        "RSR": rsr,
        "95%_CI": [ci_low, ci_high],
        "successes": successes,
        "iterations": cfg.iterations
    }

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo RSR simulator (hardware-agnostic Rowhammer).")
    parser.add_argument("--rv_ppm", type=float, default=65.0, help="Rowhammer vulnerability in ppm (e.g., A17 ≈ 65).")
    parser.add_argument("--iterations", type=int, default=100_000, help="Number of Monte Carlo trials.")
    parser.add_argument("--num_target_bits", type=int, default=20, help="Number of required bit flips.")
    parser.add_argument("--unique_offset_fraction", type=float, default=1.0, help="Fraction of unique offsets among target bits [0,1].")
    parser.add_argument("--rows_considered", type=int, default=1000, help="Effective rows explored per offset.")
    parser.add_argument("--banks_per_rank", type=int, default=16, help="Banks per rank.")
    parser.add_argument("--ranks", type=int, default=1, help="Number of ranks.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    args = parser.parse_args()

    cfg = MonteCarloConfig(
        rv_ppm=args.rv_ppm,
        iterations=args.iterations,
        num_target_bits=args.num_target_bits,
        unique_offset_fraction=args.unique_offset_fraction,
        rows_considered=args.rows_considered,
        banks_per_rank=args.banks_per_rank,
        ranks=args.ranks,
        seed=args.seed
    )
    result = simulate_rsr(cfg)
    print(json.dumps({
        "config": cfg.__dict__,
        "result": result
    }, indent=2))

if __name__ == "__main__":
    main()
