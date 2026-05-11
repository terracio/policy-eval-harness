from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from policy_eval_harness._utils.json import normalize_json_value
from policy_eval_harness._utils.manifest import (
    optional_bool,
    reject_duplicate_strings,
    reject_unknown_keys,
    require_existing_path,
)
from policy_eval_harness._utils.paths import resolve_path
from policy_eval_harness.io.tables import read_table
from policy_eval_harness.workflows.ablation_metrics import (
    bootstrap_interaction_ci,
    metric_value_from_input,
)
from policy_eval_harness.workflows.common import (
    load_mapping,
    optional_string,
    require_mapping,
    require_string,
)
from policy_eval_harness.workflows.constants import (
    ABLATION_FACTOR_EFFECTS_FILENAME,
    ABLATION_INTERACTION_SUMMARY_FILENAME,
    ABLATION_KEYS,
    ABLATION_REPORT_FILENAME,
)
from policy_eval_harness.workflows.types import (
    AblationArtifacts,
    AblationManifest,
    AblationMetricConfig,
)


def run_ablation_2x2_from_manifest(manifest_path: Path, out_dir: Path) -> AblationArtifacts:
    manifest = load_ablation_manifest(Path(manifest_path))
    frame = read_table(manifest.input_path)
    metric_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    source_kind = "panel" if {"case_id", "variant_id", "split"}.issubset(frame.columns) else "scorecard"

    for metric in manifest.metrics:
        values = {
            key: metric_value_from_input(
                frame=frame,
                source_kind=source_kind,
                split=manifest.split,
                variant_id=variant_id,
                metric_name=metric.name,
            )
            for key, variant_id in manifest.variant_map.items()
        }
        baseline_value = values["A0_B0"]
        a_effect = values["A1_B0"] - baseline_value
        b_effect = values["A0_B1"] - baseline_value
        combined_effect = values["A1_B1"] - baseline_value
        interaction_term = combined_effect - a_effect - b_effect
        oriented_interaction = interaction_term * _goal_sign(metric.goal)
        qualitative_read = _qualitative_read(oriented_interaction, metric.interaction_epsilon)
        ci_low, ci_high = bootstrap_interaction_ci(
            manifest=manifest,
            frame=frame,
            source_kind=source_kind,
            metric=metric,
        )

        metric_rows.append(
            {
                "split": manifest.split,
                "metric": metric.name,
                "metric_group": metric.group,
                "goal": metric.goal,
                "baseline_value": baseline_value,
                "a1_b0_value": values["A1_B0"],
                "a0_b1_value": values["A0_B1"],
                "a1_b1_value": values["A1_B1"],
                "a_effect": a_effect,
                "b_effect": b_effect,
                "combined_effect": combined_effect,
                "interaction_term": interaction_term,
            }
        )
        summary_rows.append(
            {
                "split": manifest.split,
                "metric": metric.name,
                "metric_group": metric.group,
                "goal": metric.goal,
                "interaction_term": interaction_term,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "oriented_interaction": oriented_interaction,
                "qualitative_read": qualitative_read,
                "interaction_epsilon": metric.interaction_epsilon,
            }
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    factor_effects_path = out_dir / ABLATION_FACTOR_EFFECTS_FILENAME
    interaction_summary_path = out_dir / ABLATION_INTERACTION_SUMMARY_FILENAME
    report_path = out_dir / ABLATION_REPORT_FILENAME

    metric_frame = pd.DataFrame(metric_rows).sort_values(["metric_group", "metric"]).reset_index(drop=True)
    summary_frame = pd.DataFrame(summary_rows).sort_values(["metric_group", "metric"]).reset_index(drop=True)
    metric_frame.to_csv(factor_effects_path, index=False)
    summary_frame.to_csv(interaction_summary_path, index=False)
    report_path.write_text(_build_ablation_report(manifest, metric_frame, summary_frame) + "\n", encoding="utf-8")

    return AblationArtifacts(
        factor_effects_path=factor_effects_path,
        interaction_summary_path=interaction_summary_path,
        report_path=report_path,
    )


def load_ablation_manifest(manifest_path: Path) -> AblationManifest:
    manifest_path = manifest_path.resolve()
    raw = load_mapping(manifest_path)
    base_dir = manifest_path.parent
    reject_unknown_keys(
        raw,
        {"input_scorecard_or_panel_path", "split", "variant_map", "metrics", "bootstrap"},
        "Ablation manifest",
    )
    raw_variant_map = require_mapping(raw, "variant_map")
    reject_unknown_keys(raw_variant_map, ABLATION_KEYS, "Manifest field 'variant_map'")
    variant_map = {key: require_string(raw_variant_map, key) for key in ABLATION_KEYS}
    reject_duplicate_strings(variant_map.values(), "Manifest field 'variant_map'")
    metrics_raw = raw.get("metrics", [])
    if not isinstance(metrics_raw, Sequence) or isinstance(metrics_raw, (str, bytes)):
        raise ValueError("Manifest field 'metrics' must be a sequence.")

    metrics: List[AblationMetricConfig] = []
    for index, item in enumerate(metrics_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"metrics[{index}] must be a mapping.")
        reject_unknown_keys(item, {"name", "goal", "group", "interaction_epsilon"}, f"metrics[{index}]")
        goal = require_string(item, "goal")
        if goal not in {"maximize", "minimize"}:
            raise ValueError("Ablation metric goal must be 'maximize' or 'minimize'.")
        metrics.append(
            AblationMetricConfig(
                name=require_string(item, "name"),
                goal=goal,
                group=require_string(item, "group"),
                interaction_epsilon=float(item.get("interaction_epsilon", 0.01) or 0.01),
            )
        )
    if not metrics:
        raise ValueError("Manifest field 'metrics' must not be empty.")
    reject_duplicate_strings((metric.name for metric in metrics), "Manifest field 'metrics.name'")

    bootstrap = raw.get("bootstrap", {})
    if bootstrap is None:
        bootstrap = {}
    if not isinstance(bootstrap, Mapping):
        raise ValueError("Manifest field 'bootstrap' must be a mapping when provided.")
    reject_unknown_keys(bootstrap, {"enabled", "n_samples", "seed"}, "Manifest field 'bootstrap'")
    bootstrap_enabled = optional_bool(bootstrap.get("enabled"), "bootstrap.enabled", default=False)
    bootstrap = dict(bootstrap)
    bootstrap["enabled"] = bootstrap_enabled
    if bootstrap_enabled and int(bootstrap.get("n_samples", 0) or 0) <= 0:
        raise ValueError("Ablation bootstrap requires 'n_samples' > 0 when enabled.")

    input_path = resolve_path(base_dir, require_string(raw, "input_scorecard_or_panel_path"))
    require_existing_path(input_path, "Manifest field 'input_scorecard_or_panel_path'")

    return AblationManifest(
        manifest_path=manifest_path,
        input_path=input_path,
        split=optional_string(raw, "split") or "holdout",
        variant_map=variant_map,
        metrics=tuple(metrics),
        bootstrap={str(key): normalize_json_value(value) for key, value in bootstrap.items()},
    )


def _build_ablation_report(
    manifest: AblationManifest,
    metric_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
) -> str:
    lines = [
        "# 2x2 Ablation Report",
        "",
        f"- Split: `{manifest.split}`",
        f"- Input: `{manifest.input_path.name}`",
        "",
        "## Variant Map",
        "",
    ]
    for key in ABLATION_KEYS:
        lines.append(f"- `{key}`: `{manifest.variant_map[key]}`")
    lines.extend(["", "## Interaction Summary", ""])
    for row in summary_frame.to_dict(orient="records"):
        lines.append(
            "- `{metric}` ({group}, {goal}): {read} "
            "(interaction={interaction:.4f}, oriented={oriented:.4f})".format(
                metric=row["metric"],
                group=row["metric_group"],
                goal=row["goal"],
                read=row["qualitative_read"],
                interaction=row["interaction_term"],
                oriented=row["oriented_interaction"],
            )
        )
    lines.extend(["", "## Factor Effects", ""])
    for row in metric_frame.to_dict(orient="records"):
        lines.append(
            "- `{metric}`: A={a:.4f}, B={b:.4f}, combined={combined:.4f}, interaction={interaction:.4f}".format(
                metric=row["metric"],
                a=row["a_effect"],
                b=row["b_effect"],
                combined=row["combined_effect"],
                interaction=row["interaction_term"],
            )
        )
    return "\n".join(lines)


def _qualitative_read(oriented_interaction: float, epsilon: float) -> str:
    if oriented_interaction > epsilon:
        return "synergy"
    if oriented_interaction < -epsilon:
        return "interference"
    return "near_additive"


def _goal_sign(goal: str) -> float:
    return 1.0 if goal == "maximize" else -1.0
