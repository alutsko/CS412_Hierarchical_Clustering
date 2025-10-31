import sys
import traceback
from pathlib import Path

# Make sure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_apriori import test_generate_and_write


def main():
    try:
        test_generate_and_write()
        print("OK: test_generate_and_write passed")
        return 0
    except AssertionError:
        print("FAIL: test_generate_and_write failed")
        traceback.print_exc()
        return 2
    except Exception:
        print("ERROR: unexpected exception while running tests")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
