from __future__ import annotations

import shutil
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path

from policy_eval_harness.public_contracts import (
    ABLATION_RELATIVE_ARTIFACTS,
    DEMO_EVALUATE_RELATIVE_ARTIFACTS,
    DEMO_REPLAY_RELATIVE_ARTIFACTS,
    LABEL_COMPARE_RELATIVE_ARTIFACTS,
    compare_artifact_bundle,
)


@dataclass(frozen=True)
class ExampleRun:
    root: Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    policy_eval = _resolve_policy_eval_executable()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = Path(tmpdir) / "public-contracts"
        _verify_demo_contract(repo_root, policy_eval, out_root / "core_demo")
        _verify_label_compare_contract(repo_root, policy_eval, out_root / "label_compare")
        _verify_ablation_contract(repo_root, policy_eval, out_root / "ablation_2x2")
    print("Public contract verification completed successfully.")
    return 0


def _resolve_policy_eval_executable() -> str:
    candidate = shutil.which("policy-eval")
    if candidate:
        return candidate

    fallback = Path(sys.executable).parent / "policy-eval"
    if fallback.exists():
        return str(fallback)

    raise RuntimeError("Could not find the installed `policy-eval` executable on PATH.")


def _verify_demo_contract(repo_root: Path, policy_eval: str, out_root: Path) -> None:
    manifest_path = repo_root / "examples" / "core_demo" / "demo.yaml"
    golden_root = repo_root / "examples" / "core_demo" / "golden"

    first = _run_public_cli(
        policy_eval,
        ["demo", "run"],
        manifest_path=manifest_path,
        out_dir=out_root / "first",
        cwd=repo_root,
    )
    second = _run_public_cli(
        policy_eval,
        ["demo", "run"],
        manifest_path=manifest_path,
        out_dir=out_root / "second",
        cwd=repo_root,
    )

    compare_artifact_bundle(first.root / "replay", golden_root / "replay", DEMO_REPLAY_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.root / "evaluate", golden_root / "evaluate", DEMO_EVALUATE_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.root / "replay", second.root / "replay", DEMO_REPLAY_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.root / "evaluate", second.root / "evaluate", DEMO_EVALUATE_RELATIVE_ARTIFACTS)


def _verify_label_compare_contract(repo_root: Path, policy_eval: str, out_root: Path) -> None:
    manifest_path = repo_root / "examples" / "label_compare" / "label_compare.yaml"
    golden_root = repo_root / "examples" / "label_compare" / "golden"

    first = _run_public_cli(
        policy_eval,
        ["workflow", "label-compare", "run"],
        manifest_path=manifest_path,
        out_dir=out_root / "first",
        cwd=repo_root,
    )
    second = _run_public_cli(
        policy_eval,
        ["workflow", "label-compare", "run"],
        manifest_path=manifest_path,
        out_dir=out_root / "second",
        cwd=repo_root,
    )

    compare_artifact_bundle(first.root, golden_root, LABEL_COMPARE_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.root, second.root, LABEL_COMPARE_RELATIVE_ARTIFACTS)


def _verify_ablation_contract(repo_root: Path, policy_eval: str, out_root: Path) -> None:
    manifest_path = repo_root / "examples" / "ablation_2x2" / "ablation_2x2.yaml"
    golden_root = repo_root / "examples" / "ablation_2x2" / "golden"

    first = _run_public_cli(
        policy_eval,
        ["workflow", "ablation-2x2", "run"],
        manifest_path=manifest_path,
        out_dir=out_root / "first",
        cwd=repo_root,
    )
    second = _run_public_cli(
        policy_eval,
        ["workflow", "ablation-2x2", "run"],
        manifest_path=manifest_path,
        out_dir=out_root / "second",
        cwd=repo_root,
    )

    compare_artifact_bundle(first.root, golden_root, ABLATION_RELATIVE_ARTIFACTS)
    compare_artifact_bundle(first.root, second.root, ABLATION_RELATIVE_ARTIFACTS)


def _run_public_cli(
    policy_eval: str,
    args: list[str],
    *,
    manifest_path: Path,
    out_dir: Path,
    cwd: Path,
) -> ExampleRun:
    subprocess.run(
        [policy_eval, *args, "--manifest", str(manifest_path), "--out-dir", str(out_dir)],
        cwd=cwd,
        check=True,
    )
    return ExampleRun(root=out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
