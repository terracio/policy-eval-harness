from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from policy_eval_harness.cli import app
from policy_eval_harness.replay import canonical_json, load_universe, run_replay_from_manifest
from policy_eval_harness.replay.runtime import normalize_json_value

CLI_RUNNER = CliRunner()


class ReplayWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_json_normalization_handles_timestamp_values(self) -> None:
        self.assertEqual(
            normalize_json_value(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
            "2026-01-01T00:00:00Z",
        )
        self.assertEqual(
            normalize_json_value(pd.Timestamp("2026-01-01T01:00:00+01:00")),
            "2026-01-01T00:00:00Z",
        )
        self.assertEqual(
            normalize_json_value(pd.Timestamp("2026-01-01T00:00:00")),
            "2026-01-01T00:00:00Z",
        )
        self.assertEqual(
            normalize_json_value(np.datetime64("2026-01-01T00:00:00.120000000")),
            "2026-01-01T00:00:00.12Z",
        )
        self.assertIsNone(normalize_json_value(pd.NaT))
        self.assertEqual(
            canonical_json({"timestamp": datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)}),
            '{"timestamp":"2026-01-01T00:00:00Z"}',
        )

    def test_load_universe_from_csv_and_parquet_matches(self) -> None:
        cases = [
            {"case_id": "case-b", "start_ts_utc": "2026-01-01T00:05:00Z", "segment": "late"},
            {"case_id": "case-a", "start_ts_utc": "2026-01-01T00:00:00Z", "segment": "early"},
        ]
        steps = [
            {"case_id": "case-b", "step_index": 1, "ts_utc": "2026-01-01T00:06:00Z", "score": 3},
            {"case_id": "case-a", "step_index": 1, "ts_utc": "2026-01-01T00:01:00Z", "score": 2},
            {"case_id": "case-a", "step_index": 0, "ts_utc": "2026-01-01T00:00:30Z", "score": 1},
        ]

        csv_cases_path, csv_steps_path = self._write_universe("csv", cases, steps)
        parquet_cases_path, parquet_steps_path = self._write_universe("parquet", cases, steps)

        csv_universe = load_universe(csv_cases_path, csv_steps_path)
        parquet_universe = load_universe(parquet_cases_path, parquet_steps_path)

        self.assertEqual(csv_universe, parquet_universe)
        self.assertEqual([case.case_id for case in csv_universe.cases], ["case-a", "case-b"])
        self.assertEqual(
            [step.step_index for step in csv_universe.steps_by_case["case-a"]],
            [0, 1],
        )

    def test_replay_run_preserves_variant_comparability(self) -> None:
        manifest_path = self._write_replay_fixture(
            cases=[
                {"case_id": "case-b", "start_ts_utc": "2026-01-01T00:05:00Z", "segment": "late"},
                {"case_id": "case-a", "start_ts_utc": "2026-01-01T00:00:00Z", "segment": "early"},
            ],
            steps=[
                {"case_id": "case-a", "step_index": 1, "ts_utc": "2026-01-01T00:01:00Z", "score": 1},
                {"case_id": "case-a", "step_index": 0, "ts_utc": "2026-01-01T00:00:30Z", "score": 0},
                {"case_id": "case-b", "step_index": 0, "ts_utc": "2026-01-01T00:06:00Z", "score": 3},
            ],
            variants=[
                {
                    "variant_id": "baseline",
                    "policy": {
                        "import_path": "tests.support.replay_fixtures:threshold_policy",
                        "params": {"threshold": 0},
                    },
                },
                {
                    "variant_id": "strict",
                    "policy": {
                        "import_path": "tests.support.replay_fixtures:threshold_policy",
                        "params": {"threshold": 2},
                    },
                },
            ],
            run_variant_ids=["baseline", "strict"],
        )

        artifacts = run_replay_from_manifest(manifest_path, self.root / "out")

        episode_summary = pd.read_csv(artifacts.episode_summary_path)
        step_trace = pd.read_parquet(artifacts.step_trace_path)

        self.assertEqual(
            episode_summary[["variant_id", "case_id"]].values.tolist(),
            [
                ["baseline", "case-a"],
                ["baseline", "case-b"],
                ["strict", "case-a"],
                ["strict", "case-b"],
            ],
        )
        self.assertEqual(
            step_trace.groupby("variant_id")["case_id"].unique().apply(list).to_dict(),
            {"baseline": ["case-a", "case-b"], "strict": ["case-a", "case-b"]},
        )
        self.assertEqual(
            step_trace[step_trace["variant_id"] == "baseline"]["step_index"].tolist(),
            [0, 1, 0],
        )
        strict_actions = step_trace[step_trace["variant_id"] == "strict"]["action_json"].tolist()
        self.assertEqual(strict_actions, ["null", "null", '{"decision":"act","score":3,"variant":"strict"}'])

    def test_cli_run_writes_artifacts_and_respects_termination(self) -> None:
        manifest_path = self._write_replay_fixture(
            cases=[
                {"case_id": "case-a", "start_ts_utc": "2026-01-01T00:00:00Z"},
            ],
            steps=[
                {"case_id": "case-a", "step_index": 0, "ts_utc": "2026-01-01T00:00:30Z", "score": 1, "stop": True},
                {"case_id": "case-a", "step_index": 1, "ts_utc": "2026-01-01T00:01:00Z", "score": 2},
            ],
            variants=[
                {
                    "variant_id": "baseline",
                    "policy": {"import_path": "tests.support.replay_fixtures:threshold_policy"},
                }
            ],
            run_variant_ids=["baseline"],
        )
        out_dir = self.root / "cli-out"

        result = CLI_RUNNER.invoke(
            app,
            ["replay", "run", "--manifest", str(manifest_path), "--out-dir", str(out_dir)],
        )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("episode_summary.csv", result.stdout)
        self.assertTrue((out_dir / "episode_summary.csv").exists())
        self.assertTrue((out_dir / "step_trace.parquet").exists())
        self.assertTrue((out_dir / "run_metadata.json").exists())

        step_trace = pd.read_parquet(out_dir / "step_trace.parquet")
        self.assertEqual(step_trace["step_index"].tolist(), [0])

        episode_summary = pd.read_csv(out_dir / "episode_summary.csv")
        self.assertEqual(episode_summary.loc[0, "termination_reason"], "stop_flag")
        self.assertTrue(bool(episode_summary.loc[0, "terminated"]))

    def test_policy_and_executor_exceptions_fail_case_and_continue(self) -> None:
        policy_manifest = self._write_replay_fixture(
            base_name="policy-error",
            cases=[
                {"case_id": "case-a", "start_ts_utc": "2026-01-01T00:00:00Z"},
                {"case_id": "case-b", "start_ts_utc": "2026-01-01T00:05:00Z"},
            ],
            steps=[
                {"case_id": "case-a", "step_index": 0, "ts_utc": "2026-01-01T00:00:30Z", "score": 1},
                {"case_id": "case-b", "step_index": 0, "ts_utc": "2026-01-01T00:05:30Z", "score": 2},
            ],
            variants=[
                {
                    "variant_id": "policy-error",
                    "policy": {
                        "import_path": "tests.support.replay_fixtures:threshold_policy",
                        "params": {"raise_on_case": "case-a"},
                    },
                }
            ],
            run_variant_ids=["policy-error"],
        )
        policy_artifacts = run_replay_from_manifest(policy_manifest, self.root / "policy-out")
        policy_summary = pd.read_csv(policy_artifacts.episode_summary_path)
        policy_trace = pd.read_parquet(policy_artifacts.step_trace_path)

        self.assertEqual(policy_summary["final_status"].tolist(), ["error", "completed"])
        self.assertEqual(policy_summary["error_count"].tolist(), [1, 0])
        self.assertTrue(policy_trace.loc[0, "error"].startswith("policy_exception:"))
        self.assertEqual(policy_trace.loc[1, "error"], "")

        executor_manifest = self._write_replay_fixture(
            base_name="executor-error",
            cases=[
                {"case_id": "case-a", "start_ts_utc": "2026-01-01T00:00:00Z"},
                {"case_id": "case-b", "start_ts_utc": "2026-01-01T00:05:00Z"},
            ],
            steps=[
                {"case_id": "case-a", "step_index": 0, "ts_utc": "2026-01-01T00:00:30Z", "score": 1},
                {"case_id": "case-b", "step_index": 0, "ts_utc": "2026-01-01T00:05:30Z", "score": 2},
            ],
            variants=[
                {
                    "variant_id": "executor-error",
                    "policy": {"import_path": "tests.support.replay_fixtures:threshold_policy"},
                }
            ],
            run_variant_ids=["executor-error"],
            executor_params={"raise_on_case": "case-b"},
        )
        executor_artifacts = run_replay_from_manifest(executor_manifest, self.root / "executor-out")
        executor_summary = pd.read_csv(executor_artifacts.episode_summary_path)
        executor_trace = pd.read_parquet(executor_artifacts.step_trace_path)

        self.assertEqual(executor_summary["final_status"].tolist(), ["completed", "error"])
        self.assertEqual(executor_summary["error_count"].tolist(), [0, 1])
        self.assertEqual(executor_trace.loc[0, "error"], "")
        self.assertTrue(executor_trace.loc[1, "error"].startswith("executor_exception:"))

    def test_replay_outputs_are_deterministic_for_same_manifest(self) -> None:
        manifest_path = self._write_replay_fixture(
            base_name="deterministic",
            cases=[
                {"case_id": "case-a", "start_ts_utc": "2026-01-01T00:00:00Z"},
                {"case_id": "case-b", "start_ts_utc": "2026-01-01T00:05:00Z"},
            ],
            steps=[
                {"case_id": "case-a", "step_index": 0, "ts_utc": "2026-01-01T00:00:30Z", "score": 1},
                {"case_id": "case-a", "step_index": 1, "ts_utc": "2026-01-01T00:01:30Z", "score": 0},
                {"case_id": "case-b", "step_index": 0, "ts_utc": "2026-01-01T00:05:30Z", "score": 2},
            ],
            variants=[
                {
                    "variant_id": "baseline",
                    "policy": {
                        "import_path": "tests.support.replay_fixtures:threshold_policy",
                        "params": {"threshold": 1},
                    },
                }
            ],
            run_variant_ids=["baseline"],
        )

        first_artifacts = run_replay_from_manifest(manifest_path, self.root / "first")
        second_artifacts = run_replay_from_manifest(manifest_path, self.root / "second")

        self.assertEqual(
            first_artifacts.episode_summary_path.read_text(encoding="utf-8"),
            second_artifacts.episode_summary_path.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            first_artifacts.run_metadata_path.read_text(encoding="utf-8"),
            second_artifacts.run_metadata_path.read_text(encoding="utf-8"),
        )
        first_trace = pd.read_parquet(first_artifacts.step_trace_path).sort_values(
            ["variant_id", "case_id", "step_index"]
        )
        second_trace = pd.read_parquet(second_artifacts.step_trace_path).sort_values(
            ["variant_id", "case_id", "step_index"]
        )
        assert_frame_equal(first_trace.reset_index(drop=True), second_trace.reset_index(drop=True))

    def _write_replay_fixture(
        self,
        *,
        cases: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        variants: List[Dict[str, Any]],
        run_variant_ids: List[str],
        base_name: str = "fixture",
        executor_params: Optional[Dict[str, Any]] = None,
    ) -> Path:
        cases_path, steps_path = self._write_universe("csv", cases, steps, base_name=base_name)
        variants_path = self.root / f"{base_name}-variants.json"
        manifest_path = self.root / f"{base_name}-manifest.json"

        variants_path.write_text(json.dumps(variants, indent=2), encoding="utf-8")
        manifest = {
            "universe": {
                "cases_path": cases_path.name,
                "steps_path": steps_path.name,
            },
            "executor": {
                "import_path": "tests.support.replay_fixtures:replay_executor",
                "params": executor_params or {},
            },
            "variants": {"path": variants_path.name},
            "run": {"variant_ids": run_variant_ids},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _write_universe(
        self,
        file_format: str,
        cases: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        *,
        base_name: str = "fixture",
    ) -> tuple[Path, Path]:
        cases_path = self.root / f"{base_name}-cases.{file_format}"
        steps_path = self.root / f"{base_name}-steps.{file_format}"

        cases_frame = pd.DataFrame(cases)
        steps_frame = pd.DataFrame(steps)
        if file_format == "csv":
            cases_frame.to_csv(cases_path, index=False)
            steps_frame.to_csv(steps_path, index=False)
        else:
            cases_frame.to_parquet(cases_path, index=False)
            steps_frame.to_parquet(steps_path, index=False)
        return cases_path, steps_path
