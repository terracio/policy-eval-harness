from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class EvaluationFixtureMixin:
    root: Path

    def _write_replay_manifest(
        self,
        *,
        summary_rows: List[Dict[str, Any]],
        cases_rows: List[Dict[str, Any]],
        gates_thresholds: Dict[str, Any] | None = None,
    ) -> Path:
        summary_path = self.root / "episode_summary.csv"
        cases_path = self.root / "cases.csv"
        manifest_path = self.root / "replay-evaluate.yaml"

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.DataFrame(cases_rows).to_csv(cases_path, index=False)

        manifest = {
            "profile": "replay_outcomes_v1",
            "inputs": {
                "episode_summary_path": summary_path.name,
                "cases_path": cases_path.name,
            },
            "baseline_variant_id": "baseline",
            "candidate_variant_ids": ["candidate"],
            "gates": {
                "profile": "sequential_promotion_v1",
                "thresholds": gates_thresholds or {},
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _write_selection_manifest(
        self,
        *,
        panel_rows: List[Dict[str, Any]],
        gates_thresholds: Dict[str, Any] | None = None,
        bootstrap: Dict[str, Any] | None = None,
        splits: Dict[str, Any] | None = None,
        include_split: bool = True,
        root: Path | None = None,
        base_name: str = "selection",
    ) -> Path:
        target_root = root or self.root
        panel_path = target_root / f"{base_name}-panel.csv"
        manifest_path = target_root / f"{base_name}-evaluate.yaml"

        frame = pd.DataFrame(panel_rows)
        if not include_split and "split" in frame.columns:
            frame = frame.drop(columns=["split"])
        frame.to_csv(panel_path, index=False)

        manifest = {
            "profile": "selection_panel_v1",
            "inputs": {
                "panel_path": panel_path.name,
            },
            "baseline_variant_id": "baseline",
            "candidate_variant_ids": ["candidate"],
            "splits": splits or {"mode": "existing"},
            "gates": {
                "profile": "selection_promotion_v1",
                "thresholds": gates_thresholds or {},
            },
        }
        if bootstrap is not None:
            manifest["bootstrap"] = bootstrap
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _summary_row(
        self,
        case_id: str,
        variant_id: str,
        utility: float | None,
        *,
        decisions: int = 1,
        terminated: bool = False,
        final_status: str = "completed",
        error_count: int = 0,
        termination_reason: str | None = None,
    ) -> Dict[str, Any]:
        summary_metrics = {} if utility is None else {"utility": utility}
        return {
            "case_id": case_id,
            "variant_id": variant_id,
            "start_ts_utc": "2026-01-01T00:00:00Z",
            "end_ts_utc": "2026-01-01T00:05:00Z",
            "n_steps": 1,
            "terminated": terminated,
            "termination_reason": termination_reason,
            "final_status": final_status,
            "decision_count": decisions,
            "error_count": error_count,
            "summary_metrics_json": json.dumps(summary_metrics, sort_keys=True),
        }

