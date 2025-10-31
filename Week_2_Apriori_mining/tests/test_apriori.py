import os
import tempfile
from pathlib import Path
from typing import Optional
from Apriori_mining import load_transactions, generate_frequent_itemsets, write_part1, write_part2


def test_generate_and_write(tmp_dir: Optional[str] = None):
    """Standalone test function. If tmp_dir is None, a temporary dir will be used.

    This mirrors the original pytest test but is runnable without pytest fixtures.
    """
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_transactions.txt")
    transactions = list(load_transactions(fixture))
    # small dataset => use min_support 0.3 (strict > threshold) so at least singletons with 2 occurrences are kept
    freq = generate_frequent_itemsets(transactions, 0.3)
    # Expect Fast Food and Restaurants to be frequent (appear 2 times)
    singletons = { next(iter(s)) for s in freq if len(s) == 1 }
    assert "Fast Food" in singletons
    assert "Restaurants" in singletons

    if tmp_dir is None:
        with tempfile.TemporaryDirectory() as d:
            _run_write_and_check(Path(d), freq)
    else:
        _run_write_and_check(Path(tmp_dir), freq)


def _run_write_and_check(path: Path, freq):
    p1 = path / "part1.txt"
    p2 = path / "part2.txt"
    write_part1(str(p1), freq)
    write_part2(str(p2), freq)
    assert p1.exists()
    assert p2.exists()

