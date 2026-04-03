from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import yaml

from policy_eval_harness.evaluation import EvaluationArtifacts, run_evaluation_from_manifest
from policy_eval_harness.replay import ReplayArtifacts, canonical_json, run_replay_from_manifest


@dataclass(frozen=True)
class DemoArtifacts:
    replay_root: Path
    evaluation_root: Path
    replay_artifacts: ReplayArtifacts
    evaluation_artifacts: EvaluationArtifacts


def load_demo_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Demo manifest must contain a mapping: {manifest_path!s}")
    return {
        "manifest_path": manifest_path,
        "replay_manifest": _resolve_path(manifest_path.parent, _require_string(payload, "replay_manifest")),
        "evaluation_manifest": _resolve_path(manifest_path.parent, _require_string(payload, "evaluation_manifest")),
        "name": payload.get("name"),
        "description": payload.get("description"),
    }


def run_demo_from_manifest(manifest_path: Path, out_dir: Path) -> DemoArtifacts:
    manifest = load_demo_manifest(manifest_path)
    out_dir = Path(out_dir).resolve()
    replay_root = out_dir / "replay"
    evaluation_root = out_dir / "evaluate"
    combined_replay_dir = replay_root / "_combined"
    evaluation_inputs_dir = evaluation_root / "inputs"
    evaluation_inputs_dir.mkdir(parents=True, exist_ok=True)

    replay_artifacts = run_replay_from_manifest(manifest["replay_manifest"], combined_replay_dir)
    _split_replay_artifacts_by_variant(replay_artifacts, replay_root)

    cases_path = _cases_path_from_replay_manifest(manifest["replay_manifest"])
    combined_summary_path = evaluation_inputs_dir / "episode_summary.csv"
    combined_summary_path.write_text(
        replay_artifacts.episode_summary_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    resolved_eval_manifest_path = evaluation_inputs_dir / "evaluate.resolved.yaml"
    _write_resolved_evaluation_manifest(
        template_path=manifest["evaluation_manifest"],
        out_path=resolved_eval_manifest_path,
        episode_summary_path=combined_summary_path,
        cases_path=cases_path,
    )
    evaluation_artifacts = run_evaluation_from_manifest(resolved_eval_manifest_path, evaluation_root)

    return DemoArtifacts(
        replay_root=replay_root,
        evaluation_root=evaluation_root,
        replay_artifacts=replay_artifacts,
        evaluation_artifacts=evaluation_artifacts,
    )


def _split_replay_artifacts_by_variant(replay_artifacts: ReplayArtifacts, replay_root: Path) -> None:
    summary = pd.read_csv(replay_artifacts.episode_summary_path)
    trace = pd.read_parquet(replay_artifacts.step_trace_path)
    metadata = json.loads(replay_artifacts.run_metadata_path.read_text(encoding="utf-8"))

    for variant_id in metadata["variant_ids"]:
        variant_dir = replay_root / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)

        variant_summary = summary[summary["variant_id"] == variant_id].reset_index(drop=True)
        variant_trace = trace[trace["variant_id"] == variant_id].reset_index(drop=True)
        variant_summary.to_csv(variant_dir / "episode_summary.csv", index=False)
        variant_trace.to_parquet(variant_dir / "step_trace.parquet", index=False)

        variant_metadata = {
            "artifact_files": metadata["artifact_files"],
            "manifest_hash": metadata["manifest_hash"],
            "package_version": metadata["package_version"],
            "universe_row_counts": metadata["universe_row_counts"],
            "variant_ids": [variant_id],
        }
        (variant_dir / "run_metadata.json").write_text(
            canonical_json(variant_metadata) + "\n",
            encoding="utf-8",
        )


def _write_resolved_evaluation_manifest(
    template_path: Path,
    out_path: Path,
    episode_summary_path: Path,
    cases_path: Path,
) -> None:
    with Path(template_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evaluation template must contain a mapping: {template_path!s}")

    payload = dict(payload)
    inputs = dict(payload.get("inputs", {}))
    inputs["episode_summary_path"] = str(episode_summary_path)
    inputs["cases_path"] = str(cases_path)
    payload["inputs"] = inputs
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _cases_path_from_replay_manifest(replay_manifest_path: Path) -> Path:
    with Path(replay_manifest_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Replay manifest must contain a mapping: {replay_manifest_path!s}")
    universe = payload.get("universe")
    if not isinstance(universe, Mapping):
        raise ValueError(f"Replay manifest {replay_manifest_path!s} is missing a universe mapping.")
    return _resolve_path(Path(replay_manifest_path).resolve().parent, _require_string(universe, "cases_path"))


def _require_string(container: Mapping[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field {key!r} must be a non-empty string.")
    return value


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()
