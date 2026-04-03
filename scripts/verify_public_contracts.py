from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from policy_eval_harness.public_contracts import verify_public_contracts

    with tempfile.TemporaryDirectory() as tmpdir:
        verify_public_contracts(repo_root, Path(tmpdir) / "public-contracts")
    print("Public contract verification completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
