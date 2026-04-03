from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from policy_eval_harness.cli import app

CLI_RUNNER = CliRunner()


class DemoWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.repo_root = Path(__file__).resolve().parents[1]
        self.demo_root = self.repo_root / "examples" / "core_demo"
        self.golden_root = self.demo_root / "golden"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_demo_cli_runs_and_emits_expected_verdicts(self) -> None:
        out_dir = self.root / "demo-run"
        manifest = self.demo_root / "demo.yaml"

        result = CLI_RUNNER.invoke(
            app,
            ["demo", "run", "--manifest", str(manifest), "--out-dir", str(out_dir)],
        )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertTrue((out_dir / "replay" / "approval_targeted_v2" / "episode_summary.csv").exists())
        self.assertTrue((out_dir / "evaluate" / "scorecard.csv").exists())

        decisions = json.loads((out_dir / "evaluate" / "promotion_decisions.json").read_text(encoding="utf-8"))
        verdicts = {item["candidate_variant_id"]: item["verdict"] for item in decisions["candidates"]}
        self.assertEqual(verdicts["approval_targeted_v2"], "pass")
        self.assertEqual(verdicts["approval_overactive_v1"], "fail")

    def test_demo_outputs_match_checked_in_goldens(self) -> None:
        out_dir = self.root / "demo-run"
        manifest = self.demo_root / "demo.yaml"

        result = CLI_RUNNER.invoke(
            app,
            ["demo", "run", "--manifest", str(manifest), "--out-dir", str(out_dir)],
        )
        self.assertEqual(result.exit_code, 0, result.stdout)

        expected_files = [
            Path("evaluate/comparison_panel.parquet"),
            Path("evaluate/inputs/episode_summary.csv"),
            Path("evaluate/promotion_decisions.json"),
            Path("evaluate/scorecard.csv"),
            Path("replay/_combined/episode_summary.csv"),
            Path("replay/_combined/run_metadata.json"),
            Path("replay/_combined/step_trace.parquet"),
            Path("replay/approval_baseline_v1/episode_summary.csv"),
            Path("replay/approval_baseline_v1/run_metadata.json"),
            Path("replay/approval_baseline_v1/step_trace.parquet"),
            Path("replay/approval_overactive_v1/episode_summary.csv"),
            Path("replay/approval_overactive_v1/run_metadata.json"),
            Path("replay/approval_overactive_v1/step_trace.parquet"),
            Path("replay/approval_targeted_v2/episode_summary.csv"),
            Path("replay/approval_targeted_v2/run_metadata.json"),
            Path("replay/approval_targeted_v2/step_trace.parquet"),
        ]

        for relative_path in expected_files:
            actual_path = out_dir / relative_path
            golden_path = self.golden_root / relative_path
            self.assertTrue(actual_path.exists(), str(actual_path))
            self.assertTrue(golden_path.exists(), str(golden_path))
            self._assert_file_matches(actual_path, golden_path)

    def test_replay_and_evaluate_manifests_run_separately(self) -> None:
        replay_out_dir = self.root / "separate-replay"
        evaluate_out_dir = self.root / "separate-evaluate"

        replay_result = CLI_RUNNER.invoke(
            app,
            ["replay", "run", "--manifest", str(self.demo_root / "replay.yaml"), "--out-dir", str(replay_out_dir)],
        )
        self.assertEqual(replay_result.exit_code, 0, replay_result.stdout)
        self.assertTrue((replay_out_dir / "episode_summary.csv").exists())

        evaluate_result = CLI_RUNNER.invoke(
            app,
            ["evaluate", "run", "--manifest", str(self.demo_root / "evaluate.yaml"), "--out-dir", str(evaluate_out_dir)],
        )
        self.assertEqual(evaluate_result.exit_code, 0, evaluate_result.stdout)

        decisions = json.loads(
            (evaluate_out_dir / "promotion_decisions.json").read_text(encoding="utf-8")
        )
        verdicts = {item["candidate_variant_id"]: item["verdict"] for item in decisions["candidates"]}
        self.assertEqual(verdicts["approval_targeted_v2"], "pass")
        self.assertEqual(verdicts["approval_overactive_v1"], "fail")

    def _assert_file_matches(self, actual_path: Path, expected_path: Path) -> None:
        suffix = actual_path.suffix.lower()
        if suffix == ".json":
            self.assertEqual(
                json.loads(actual_path.read_text(encoding="utf-8")),
                json.loads(expected_path.read_text(encoding="utf-8")),
            )
            return
        if suffix == ".csv":
            assert_frame_equal(
                self._normalized_frame(pd.read_csv(actual_path)),
                self._normalized_frame(pd.read_csv(expected_path)),
                check_dtype=False,
            )
            return
        if suffix == ".parquet":
            assert_frame_equal(
                self._normalized_frame(pd.read_parquet(actual_path)),
                self._normalized_frame(pd.read_parquet(expected_path)),
                check_dtype=False,
            )
            return
        self.fail(f"Unsupported artifact type for comparison: {actual_path}")

    def _normalized_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.reset_index(drop=True)
        return frame.sort_values(list(frame.columns)).reset_index(drop=True)
