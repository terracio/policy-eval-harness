from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from policy_eval_harness.evaluation import load_evaluation_manifest
from policy_eval_harness.replay import load_replay_manifest
from policy_eval_harness.workflows import load_ablation_manifest, load_label_compare_manifest
from tests.support.evaluation_fixtures import EvaluationFixtureMixin


class ManifestValidationTests(EvaluationFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_evaluation_rejects_unknown_threshold_keys(self) -> None:
        manifest_path = self._write_selection_manifest(
            panel_rows=[
                {
                    "case_id": "case-a",
                    "variant_id": "baseline",
                    "split": "holdout",
                    "selected": True,
                    "label": True,
                    "utility": 1.0,
                },
                {
                    "case_id": "case-a",
                    "variant_id": "candidate",
                    "split": "holdout",
                    "selected": True,
                    "label": True,
                    "utility": 1.0,
                },
            ],
            gates_thresholds={"min_delta_balanced_accuracy_typo": 0.0},
        )

        with self.assertRaisesRegex(ValueError, "gates.thresholds contains unsupported fields"):
            load_evaluation_manifest(manifest_path)

    def test_evaluation_string_false_bootstrap_is_disabled(self) -> None:
        manifest_path = self._write_selection_manifest(
            panel_rows=[
                {
                    "case_id": "case-a",
                    "variant_id": "baseline",
                    "split": "holdout",
                    "selected": True,
                    "label": True,
                    "utility": 1.0,
                },
                {
                    "case_id": "case-a",
                    "variant_id": "candidate",
                    "split": "holdout",
                    "selected": True,
                    "label": True,
                    "utility": 1.0,
                },
            ],
            bootstrap={"enabled": "false"},
        )

        manifest = load_evaluation_manifest(manifest_path)
        self.assertFalse(manifest.bootstrap.enabled)

    def test_replay_rejects_unknown_variant_fields(self) -> None:
        cases_path = self.root / "cases.csv"
        steps_path = self.root / "steps.csv"
        variants_path = self.root / "variants.json"
        manifest_path = self.root / "replay.json"
        pd.DataFrame([{"case_id": "case-a"}]).to_csv(cases_path, index=False)
        pd.DataFrame([{"case_id": "case-a", "step_index": 0, "ts_utc": "2026-01-01T00:00:00Z"}]).to_csv(
            steps_path,
            index=False,
        )
        variants_path.write_text(
            json.dumps(
                [
                    {
                        "variant_id": "baseline",
                        "policy": {"import_path": "tests.support.replay_fixtures:baseline_policy"},
                        "typo": "ignored before validation",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "universe": {"cases_path": cases_path.name, "steps_path": steps_path.name},
                    "executor": {"import_path": "tests.support.replay_fixtures:replay_executor"},
                    "variants": {"path": variants_path.name},
                    "run": {"variant_ids": ["baseline"]},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Variant entry contains unsupported fields: typo"):
            load_replay_manifest(manifest_path)

    def test_workflow_manifests_reject_unknown_fields(self) -> None:
        dataset_path = self.root / "dataset.csv"
        dataset_path.write_text("sample_id,split,feature,label\n1,train,0.1,true\n", encoding="utf-8")
        label_manifest = self.root / "label.yaml"
        label_manifest.write_text(
            yaml.safe_dump(
                {
                    "dataset": {
                        "path": dataset_path.name,
                        "id_column": "sample_id",
                        "split_column": "split",
                        "feature_columns": ["feature"],
                        "label_variants": ["label"],
                        "unexpected": "field",
                    },
                    "models": ["random_forest"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Manifest field 'dataset' contains unsupported fields: unexpected"):
            load_label_compare_manifest(label_manifest)

        panel_path = self.root / "panel.csv"
        panel_path.write_text("case_id,variant_id,split,utility\ncase-a,A0,holdout,1.0\n", encoding="utf-8")
        ablation_manifest = self.root / "ablation.yaml"
        ablation_manifest.write_text(
            yaml.safe_dump(
                {
                    "input_scorecard_or_panel_path": panel_path.name,
                    "variant_map": {"A0_B0": "A0", "A1_B0": "A1", "A0_B1": "B1", "A1_B1": "AB"},
                    "metrics": [{"name": "mean_utility", "group": "utility", "goal": "maximize", "typo": 1}],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "metrics\\[0\\] contains unsupported fields: typo"):
            load_ablation_manifest(ablation_manifest)
