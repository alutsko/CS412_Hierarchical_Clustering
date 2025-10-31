#!/usr/bin/env python3
"""
Contiguous sequential pattern miner.

Usage:
  python3 contiguous_seq_miner.py --input reviews_sample.txt --min-support 0.01 --output patterns.txt

This implements an iterative contiguous-extension algorithm: start from frequent 1-item patterns,
then extend frequent k-item contiguous patterns by one token to the right when they appear in a
sequence, counting each sequence at most once per candidate. Writes results as lines
  support:item1;item2;... 
with absolute support >= threshold.
"""
import argparse
import math
from collections import defaultdict
from typing import List, Tuple


def load_sequences(path: str) -> List[List[str]]:
    sequences = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                sequences.append([])
            else:
                # tokens are whitespace-separated; keep them as-is
                tokens = line.split()
                sequences.append(tokens)
    return sequences


def mine_contiguous_patterns(sequences: List[List[str]], min_support_abs: int):
    n_seq = len(sequences)
    # count 1-item supports (sequence-level support: count at most once per sequence)
    single_counts = defaultdict(int)
    for seq in sequences:
        seen = set(seq)
        for token in seen:
            single_counts[(token,)] += 1

    # keep patterns meeting threshold
    frequent = {}
    L1 = {pat: cnt for pat, cnt in single_counts.items() if cnt >= min_support_abs}
    frequent.update(L1)

    print(f"Sequences: {n_seq}; min_support_abs: {min_support_abs}; frequent 1-item: {len(L1)}")

    current_freq = L1  # mapping tuple -> count
    k = 1
    # iteratively build length-(k+1) patterns
    while current_freq:
        candidates_counts = defaultdict(int)
        current_patterns_set = set(current_freq.keys())

        for seq in sequences:
            m = len(seq)
            if m <= k:
                continue
            seen_in_seq = set()
            # scan contiguous windows of length k, extend to k+1
            # i from 0..m-k-1 to allow access tokens[i+k]
            for i in range(0, m - k):
                window = tuple(seq[i:i + k])
                if window in current_patterns_set:
                    # extend to the right by one token
                    cand = window + (seq[i + k],)
                    seen_in_seq.add(cand)
            for cand in seen_in_seq:
                candidates_counts[cand] += 1

        # filter candidates by threshold
        next_freq = {pat: cnt for pat, cnt in candidates_counts.items() if cnt >= min_support_abs}
        if not next_freq:
            break
        frequent.update(next_freq)
        k += 1
        current_freq = next_freq
        print(f"Found {len(next_freq)} frequent patterns of length {k}.")

    return frequent


def write_patterns(path: str, patterns: dict):
    # patterns: dict of tuple->count
    # sort by descending support then by length then lexicographically for determinism
    items = sorted(patterns.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    with open(path, "w", encoding="utf-8") as f:
        for pat, cnt in items:
            line = f"{cnt}:{';'.join(pat)}\n"
            f.write(line)


def parse_min_support(value: str, n_seq: int) -> int:
    # if value is <=1 treat as relative ratio; else absolute
    v = float(value)
    if v <= 1.0:
        return max(1, math.ceil(v * n_seq))
    else:
        return int(v)


def main():
    parser = argparse.ArgumentParser(description="Mine contiguous sequential patterns")
    parser.add_argument("--input", required=True, help="Input reviews file (one sequence per line)")
    parser.add_argument("--min-support", default="0.01",
                        help="Minimum support: relative (0-1) or absolute (>=1). Default 0.01")
    parser.add_argument("--output", default="patterns.txt", help="Output file path")
    args = parser.parse_args()

    sequences = load_sequences(args.input)
    n = len(sequences)
    min_support_abs = parse_min_support(args.min_support, n)

    frequent = mine_contiguous_patterns(sequences, min_support_abs)
    write_patterns(args.output, frequent)
    print(f"Wrote {len(frequent)} patterns to {args.output}")


if __name__ == "__main__":
    main()
