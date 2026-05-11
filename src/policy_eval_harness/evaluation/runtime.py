from __future__ import annotations

from pathlib import Path

from policy_eval_harness.evaluation.constants import (
    COMPARISON_PANEL_FILENAME,
    PROFILE_REPLAY,
    PROFILE_SELECTION,
    PROMOTION_DECISIONS_FILENAME,
    SCORECARD_FILENAME,
)
from policy_eval_harness.evaluation.manifest import load_evaluation_manifest
from policy_eval_harness.evaluation.panels import build_replay_panel, build_selection_panel
from policy_eval_harness.evaluation.replay_profile import evaluate_replay_profile
from policy_eval_harness.evaluation.selection_profile import evaluate_selection_profile
from policy_eval_harness.evaluation.types import EvaluationArtifacts
from policy_eval_harness.replay import canonical_json


def run_evaluation_from_manifest(manifest_path: Path, out_dir: Path) -> EvaluationArtifacts:
    manifest = load_evaluation_manifest(Path(manifest_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest.profile_id == PROFILE_REPLAY:
        panel = build_replay_panel(manifest)
        scorecard, decisions = evaluate_replay_profile(manifest, panel)
    elif manifest.profile_id == PROFILE_SELECTION:
        panel = build_selection_panel(manifest)
        scorecard, decisions = evaluate_selection_profile(manifest, panel)
    else:
        raise ValueError(f"Unsupported evaluation profile: {manifest.profile_id!r}")

    panel = panel.sort_values(["split", "variant_id", "case_id"]).reset_index(drop=True)
    scorecard = scorecard.sort_values(["split", "candidate_variant_id", "metric"]).reset_index(drop=True)

    comparison_panel_path = out_dir / COMPARISON_PANEL_FILENAME
    scorecard_path = out_dir / SCORECARD_FILENAME
    promotion_decisions_path = out_dir / PROMOTION_DECISIONS_FILENAME

    panel.to_parquet(comparison_panel_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    promotion_decisions_path.write_text(canonical_json(decisions) + "\n", encoding="utf-8")

    return EvaluationArtifacts(
        comparison_panel_path=comparison_panel_path,
        scorecard_path=scorecard_path,
        promotion_decisions_path=promotion_decisions_path,
    )

