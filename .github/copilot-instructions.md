
# Copilot instructions for this workspace

This repository is a minimal, single-script Apriori mining project. The goal of this file is to give an AI coding agent the concrete, repo-specific guidance needed to be immediately productive.

## Big picture
- Single primary script: `Apriori_mining.py` (root). Treat this as the canonical place for algorithm logic, helpers, and the CLI entrypoint.
- Example data file (used only as an example of naming and format): `A-NUHa0sQ1C0nBpyRUTG5g_3260c54dad114a1a96af95bf746f9df1_categories.txt`
- There is no package structure, no dependency manifest, and no test harness present by default.

## What to expect and why this layout
- Small teaching/demo project: one file keeps algorithm and I/O close together for easy inspection.
- Avoid adding large raw data to the repo root. Use `data/` or `tests/fixtures/` for small reproducible inputs.

## Conventions & patterns to follow in edits
- Factor logic into small functions inside `Apriori_mining.py`. Suggested helpers (used as examples throughout the repo):
	- `load_transactions(path) -> Iterator[List[str]]` — yield each transaction (list of items)
	- `generate_frequent_itemsets(transactions, min_support) -> Dict[frozenset, int]`
	- `write_output(path, results)` — write results in a stable, machine-readable format (CSV/TSV/JSON)
- Provide a module-level `if __name__ == "__main__":` block that parses flags with `argparse` and calls the above functions. This makes the script importable for unit tests.
- Preserve UTF-8 encoding and any top-of-file comments when editing `Apriori_mining.py`.

## CLI / developer workflows (what works now / expected)
- Run the script directly (no virtualenv or deps currently required):
	- `python3 Apriori_mining.py --input <transactions-file> --min-support 0.05 --output results.json`
- If you add third-party packages, add a `requirements.txt` and document install steps in `README.md`.
- Tests: none exist. Use `pytest` if you add tests. Place tests in `tests/` and fixtures in `tests/fixtures/`.

## Project-specific advice for AI agents
- Because the code is in a single file, prefer small, local changes that add functions and keep the external API stable.
- When adding new files (tests, README, requirements), update `README.md` and add a one-line note in `requirements.txt` explaining why the dependency was added.
- Do not commit large dataset files to the root. If you need an example dataset, create `tests/fixtures/sample_transactions.txt` (small, human-readable) instead.

## Integration points & performance notes
- No external services or network calls are present. The main integration point is filesystem I/O for input transaction files and result outputs.
- Transactions may be large. Prefer streaming (generator-based) transaction readers and incremental counting to avoid high memory usage.

## Small contract to follow when changing behavior
- Inputs: path to a text file of transactions (one transaction per line, items separated by whitespace or commas).
- Outputs: file path containing frequent itemsets and support counts (machine-readable: JSON/CSV).
- Error modes: file not found, malformed lines — handle by logging/skipping with a clear message and non-zero exit code for fatal errors.

## Edge cases to check in tests (suggested)
- Empty or missing input file
- Non-UTF8 bytes in input
- Very large transactions (memory pressure)
- min_support set to 0 or 1 (boundary behavior)

## When merging or updating this file
- If an existing `.github/copilot-instructions.md` exists, preserve any repo-specific examples and update only the parts that are no longer accurate (for example, added files or tests). The current repository has the single `Apriori_mining.py` script and the example data file listed above.

## Quick checklist for PRs from an AI agent
- Add or update small, well-scoped functions in `Apriori_mining.py`.
- Add tests under `tests/` with small fixtures under `tests/fixtures/`.
- If new third-party packages are added: add `requirements.txt` and mention the package in `README.md`.
- Run `python3 -m pytest` locally in CI or as part of the PR.

If any part of the repo is unclear or you'd like me to expand this with concrete examples (argparse skeleton, test scaffold, or a `README.md`), tell me which piece to add and I'll implement it.


