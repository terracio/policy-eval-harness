from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from pandas.testing import assert_frame_equal

from policy_eval_harness.demo import run_demo_from_manifest
from policy_eval_harness.workflows import (
    run_ablation_2x2_from_manifest,
    run_label_compare_from_manifest,
)

DEMO_REPLAY_RELATIVE_ARTIFACTS: tuple[Path, ...] = (
    Path("_combined/episode_summary.csv"),
    Path("_combined/run_metadata.json"),
    Path("_combined/step_trace.parquet"),
    Path("approval_baseline_v1/episode_summary.csv"),
    Path("approval_baseline_v1/run_metadata.json"),
    Path("approval_baseline_v1/step_trace.parquet"),
    Path("approval_overactive_v1/episode_summary.csv"),
    Path("approval_overactive_v1/run_metadata.json"),
    Path("approval_overactive_v1/step_trace.parquet"),
    Path("approval_targeted_v2/episode_summary.csv"),
    Path("approval_targeted_v2/run_metadata.json"),
    Path("approval_targeted_v2/step_trace.parquet"),
)

DEMO_EVALUATE_RELATIVE_ARTIFACTS: tuple[Path, ...] = (
    Path("comparison_panel.parquet"),
    Path("inputs/cases.csv"),
    Path("inputs/episode_summary.csv"),
    Path("inputs/evaluate.resolved.yaml"),
    Path("promotion_decisions.json"),
    Path("scorecard.csv"),
)

LABEL_COMPARE_RELATIVE_ARTIFACTS: tuple[Path, ...] = (
    Path("oos_auc_by_label.csv"),
    Path("model_label_summary.json"),
    Path("feature_manifest_snapshot.json"),
)

ABLATION_RELATIVE_ARTIFACTS: tuple[Path, ...] = (
    Path("factor_effects.csv"),
    Path("interaction_summary.csv"),
    Path("ablation_report.md"),
)


def verify_public_contracts(repo_root: Path, out_root: Path) -> None:
    repo_root = Path(repo_root).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    _verify_demo_contract(repo_root, out_root / "core_demo")
    _verify_label_compare_contract(repo_root, out_root / "label_compare")
    _verify_ablation_contract(repo_root, out_root / "ablation_2x2")


def _verify_demo_contract(repo_root: Path, out_root: Path) -> None:
    manifest_path = repo_root / "examples" / "core_demo" / "demo.yaml"
    golden_root = repo_root / "examples" / "core_demo" / "golden"

    first = run_demo_from_manifest(manifest_path, out_root / "first")
    second = run_demo_from_manifest(manifest_path, out_root / "second")

    compare_artifact_bundle(first.replay_root, golden_root / "replay", DEMO_REPLAY_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.evaluation_root, golden_root / "evaluate", DEMO_EVALUATE_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.replay_root, second.replay_root, DEMO_REPLAY_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.evaluation_root, second.evaluation_root, DEMO_EVALUATE_RELATIVE_ARTIFACTS)


def _verify_label_compare_contract(repo_root: Path, out_root: Path) -> None:
    manifest_path = repo_root / "examples" / "label_compare" / "label_compare.yaml"
    golden_root = repo_root / "examples" / "label_compare" / "golden"

    first = run_label_compare_from_manifest(manifest_path, out_root / "first")
    second = run_label_compare_from_manifest(manifest_path, out_root / "second")

    compare_artifact_bundle(first.oos_auc_path.parent, golden_root, LABEL_COMPARE_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.oos_auc_path.parent, second.oos_auc_path.parent, LABEL_COMPARE_RELATIVE_ARTIFACTS)


def _verify_ablation_contract(repo_root: Path, out_root: Path) -> None:
    manifest_path = repo_root / "examples" / "ablation_2x2" / "ablation_2x2.yaml"
    golden_root = repo_root / "examples" / "ablation_2x2" / "golden"

    first = run_ablation_2x2_from_manifest(manifest_path, out_root / "first")
    second = run_ablation_2x2_from_manifest(manifest_path, out_root / "second")

    compare_artifact_bundle(first.factor_effects_path.parent, golden_root, ABLATION_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.factor_effects_path.parent, second.factor_effects_path.parent, ABLATION_RELATIVE_ARTIFACTS)


def compare_artifact_bundle(
    actual_root: Path,
    expected_root: Path,
    relative_paths: Sequence[Path],
    *,
    require_exact: bool = True,
) -> None:
    actual_root = Path(actual_root)
    expected_root = Path(expected_root)
    contract_paths = tuple(Path(path) for path in relative_paths)
    if require_exact:
        _assert_exact_artifact_set(actual_root, contract_paths, "actual")
        _assert_exact_artifact_set(expected_root, contract_paths, "expected")
    for relative_path in contract_paths:
        compare_artifact_file(actual_root / relative_path, expected_root / relative_path)


def compare_artifact_file(actual_path: Path, expected_path: Path) -> None:
    actual_path = Path(actual_path)
    expected_path = Path(expected_path)

    if not actual_path.exists():
        raise AssertionError(f"Missing artifact: {actual_path}")
    if not expected_path.exists():
        raise AssertionError(f"Missing expected artifact: {expected_path}")

    suffix = actual_path.suffix.lower()
    if suffix == ".json":
        if _load_json(actual_path) != _load_json(expected_path):
            raise AssertionError(f"JSON artifact mismatch: {actual_path} != {expected_path}")
        return
    if suffix == ".csv":
        assert_frame_equal(
            _normalized_frame(pd.read_csv(actual_path)),
            _normalized_frame(pd.read_csv(expected_path)),
            check_dtype=False,
        )
        return
    if suffix == ".parquet":
        assert_frame_equal(
            _normalized_frame(pd.read_parquet(actual_path)),
            _normalized_frame(pd.read_parquet(expected_path)),
            check_dtype=False,
        )
        return
    if suffix == ".md":
        if actual_path.read_text(encoding="utf-8") != expected_path.read_text(encoding="utf-8"):
            raise AssertionError(f"Markdown artifact mismatch: {actual_path} != {expected_path}")
        return
    if suffix in {".yaml", ".yml"}:
        if _load_yaml(actual_path) != _load_yaml(expected_path):
            raise AssertionError(f"YAML artifact mismatch: {actual_path} != {expected_path}")
        return

    raise ValueError(f"Unsupported artifact type for comparison: {actual_path}")


def _assert_exact_artifact_set(root: Path, relative_paths: Sequence[Path], label: str) -> None:
    observed = _relative_file_set(root)
    expected = {Path(path) for path in relative_paths}
    unexpected = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unexpected or missing:
        parts = []
        if unexpected:
            parts.append("unexpected: " + ", ".join(str(path) for path in unexpected))
        if missing:
            parts.append("missing: " + ", ".join(str(path) for path in missing))
        raise AssertionError(f"{label.title()} artifact set mismatch for {root}: {'; '.join(parts)}")


def _relative_file_set(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    sort_key = frame.astype(str).agg("\u001f".join, axis=1)
    return (
        frame.assign(__sort_key__=sort_key)
        .sort_values("__sort_key__", kind="mergesort")
        .drop(columns="__sort_key__")
        .reset_index(drop=True)
    )
