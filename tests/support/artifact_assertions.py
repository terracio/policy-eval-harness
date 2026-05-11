from __future__ import annotations

from pathlib import Path
from typing import Sequence

from policy_eval_harness.public_contracts import compare_artifact_bundle


def assert_artifact_bundle_matches(testcase, actual_root: Path, expected_root: Path, relative_paths: Sequence[Path]) -> None:
    for relative_path in relative_paths:
        testcase.assertTrue((actual_root / relative_path).exists(), str(actual_root / relative_path))
        testcase.assertTrue((expected_root / relative_path).exists(), str(expected_root / relative_path))
    compare_artifact_bundle(actual_root, expected_root, relative_paths)
