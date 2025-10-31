#!/usr/bin/env python3
"""Apriori_mining.py

Small, single-file Apriori helper. This edit implements Part 1 of the
assignment: count length-1 (singleton) categories and write `part1.txt`.

Usage (examples):
  python3 Apriori_mining.py --input A-NUHa0sQ1C0nBpyRUTG5g_3260c54dad114a1a96af95bf746f9df1_categories.txt \
	--min-support 0.01 --output-part1 part1.txt

The script computes the absolute support threshold as ceil(min_support * N)
so that "relative min support 0.01 on N=77185" -> ceil(771.85) == 772 (match
the assignment requirement: supports larger than 771).
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from typing import Dict, Iterable, List, Tuple


def parse_transaction_line(line: str) -> List[str]:
	"""Parse one line of the input file into a list of category strings.

	- Splits on semicolon (';').
	- Strips whitespace around category names.
	- Deduplicates categories inside a single transaction (treat transaction as a set).
	- Skips empty tokens.
	"""
	parts = [p.strip() for p in line.split(";")]
	parts = [p for p in parts if p]
	# Deduplicate within transaction to avoid double-counting the same category in one place
	return list(dict.fromkeys(parts))


def count_singletons(input_path: str) -> Tuple[collections.Counter, int]:
	"""Stream the input file and count single-item supports.

	Returns a Counter mapping category -> absolute support and the total
	number of transactions (N).
	"""
	counter = collections.Counter()
	total = 0
	try:
		with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
			for raw in fh:
				line = raw.strip()
				if not line:
					continue
				total += 1
				items = parse_transaction_line(line)
				# Increment once per transaction per item
				counter.update(items)
	except FileNotFoundError:
		print(f"Input file not found: {input_path}", file=sys.stderr)
		raise
	return counter, total


def write_part1(output_path: str, counts: Dict[str, int], threshold: int) -> None:
	"""Write frequent singletons to `output_path` using format `support:Category`.

	Only items with support >= threshold are written. Lines are sorted by
	descending support then lexicographically by category for determinism.
	"""
	# Filter
	items = [(support, cat) for cat, support in counts.items() if support >= threshold]
	# Sort by support desc, then category asc
	items.sort(key=lambda x: (-x[0], x[1]))
	with open(output_path, "w", encoding="utf-8") as fh:
		for support, cat in items:
			fh.write(f"{support}:{cat}\n")


def apriori_gen(prev_frequents: List[frozenset]) -> List[frozenset]:
	"""Generate candidate k-itemsets (as frozensets) from frequent (k-1)-itemsets.

	Follows the standard join-and-prune Apriori method: join itemsets that
	share the first k-2 items (when represented as sorted tuples), then prune
	candidates that contain any (k-1)-subset not present in prev_frequents.
	"""
	candidates = []
	prev_set = set(prev_frequents)
	if not prev_frequents:
		return candidates
	k = len(next(iter(prev_frequents))) + 1
	# Prepare sorted tuples for deterministic join
	tuples = [tuple(sorted(x)) for x in prev_frequents]
	tuples.sort()
	n = len(tuples)
	for i in range(n):
		for j in range(i + 1, n):
			a, b = tuples[i], tuples[j]
			# join if first k-2 items are equal
			if a[:-1] == b[:-1]:
				cand = frozenset(a + (b[-1],))
				# prune: all (k-1)-subsets of cand must be in prev_set
				ok = True
				for subset in __import__("itertools").combinations(cand, k - 1):
					if frozenset(subset) not in prev_set:
						ok = False
						break
				if ok:
					candidates.append(cand)
			else:
				break
	return candidates


def count_candidates(input_path: str, candidates: List[frozenset]) -> Dict[frozenset, int]:
	"""Count support for the given candidate itemsets by streaming the file.

	For each transaction, generate combinations of the transaction of size k
	and check membership in the candidate set. Returns mapping cand->support.
	"""
	from itertools import combinations

	counts: Dict[frozenset, int] = {c: 0 for c in candidates}
	cand_set = set(candidates)
	if not candidates:
		return counts
	k = len(next(iter(candidates)))
	with open(input_path, "r", encoding="utf-8", errors="replace") as fh:
		for raw in fh:
			line = raw.strip()
			if not line:
				continue
			items = parse_transaction_line(line)
			if len(items) < k:
				continue
			# generate combinations of size k from this transaction
			for comb in combinations(items, k):
				fs = frozenset(comb)
				if fs in cand_set:
					counts[fs] += 1
	return counts


def generate_frequent_itemsets(input_path: str, min_support: float) -> Dict[frozenset, int]:
	"""Run the Apriori algorithm and return a map of frequent itemsets->support.

	The absolute threshold is computed as ceil(min_support * N).
	"""
	# First pass: count singletons and get N
	single_counts, total = count_singletons(input_path)
	if total == 0:
		return {}
	threshold = math.ceil(min_support * total)
	# Frequent 1-itemsets
	frequents: Dict[frozenset, int] = {}
	L1 = [frozenset([item]) for item, sup in single_counts.items() if sup >= threshold]
	# store supports
	for item in L1:
		frequents[item] = single_counts[next(iter(item))]

	k = 2
	Lk_minus_1 = L1
	while Lk_minus_1:
		# generate candidates Ck
		Ck = apriori_gen(Lk_minus_1)
		if not Ck:
			break
		# count supports for Ck
		counts = count_candidates(input_path, Ck)
		# filter by threshold
		Lk = [c for c, s in counts.items() if s >= threshold]
		for c in Lk:
			frequents[c] = counts[c]
		Lk_minus_1 = Lk
		k += 1

	return frequents


def write_part2(output_path: str, frequents: Dict[frozenset, int]) -> None:
	"""Write all frequent itemsets to output_path in format `support:cat1;cat2;...`.

	Item order inside a set is sorted alphabetically for determinism.
	Lines are sorted by itemset length ascending, then support descending.
	"""
	# Prepare list of (support, sorted_items_list)
	rows = []
	for itemset, sup in frequents.items():
		items = sorted(itemset)
		rows.append((len(items), -sup, sup, items))
	rows.sort()
	with open(output_path, "w", encoding="utf-8") as fh:
		for _, __, sup, items in rows:
			fh.write(f"{sup}:{';'.join(items)}\n")


def main(argv: List[str]) -> int:
	parser = argparse.ArgumentParser(description="Apriori Part 1: singleton counts")
	parser.add_argument("--input", required=True, help="Path to transactions file")
	parser.add_argument("--min-support", required=True, type=float,
						help="Relative minimum support (e.g. 0.01)")
	parser.add_argument("--output-part1", required=True, help="Path to write part1.txt")
	parser.add_argument("--output-part2", required=False, help="Path to write part2.txt")
	args = parser.parse_args(argv)

	counts, total = count_singletons(args.input)
	if total == 0:
		print("No transactions found in input file.", file=sys.stderr)
		return 2

	# Use ceiling to match assignment semantics: absolute support strictly larger
	# than floor(min_support * N) -> ceil(min_support * N)
	threshold = math.ceil(args.min_support * total)
	print(f"Transactions: {total}; min_support (rel): {args.min_support}; threshold (abs): {threshold}")

	write_part1(args.output_part1, counts, threshold)
	print(f"Wrote single-item frequent categories to {args.output_part1}")

	if args.output_part2:
		print("Generating all frequent itemsets (this may take some time)...")
		frequents = generate_frequent_itemsets(args.input, args.min_support)
		write_part2(args.output_part2, frequents)
		print(f"Wrote all frequent itemsets to {args.output_part2} (count={len(frequents)})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))

