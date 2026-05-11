from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from policy_eval_harness.public_contracts import (
    ABLATION_RELATIVE_ARTIFACTS,
    DEMO_EVALUATE_RELATIVE_ARTIFACTS,
    DEMO_REPLAY_RELATIVE_ARTIFACTS,
    LABEL_COMPARE_RELATIVE_ARTIFACTS,
    compare_artifact_bundle,
    verify_public_contracts,
)
from tests.support.artifact_assertions import assert_artifact_bundle_matches


class PublicContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.repo_root = Path(__file__).resolve().parents[1]

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_public_contract_verification_runs_end_to_end(self) -> None:
        verify_public_contracts(self.repo_root, self.root / "verification")

    def test_public_contract_rejects_unexpected_artifacts(self) -> None:
        actual_root = self.root / "actual"
        expected_root = self.root / "expected"
        actual_root.mkdir()
        expected_root.mkdir()
        (actual_root / "artifact.json").write_text('{"ok": true}\n', encoding="utf-8")
        (expected_root / "artifact.json").write_text('{"ok": true}\n', encoding="utf-8")
        (actual_root / "extra.json").write_text('{"extra": true}\n', encoding="utf-8")

        with self.assertRaisesRegex(AssertionError, "unexpected: extra.json"):
            compare_artifact_bundle(actual_root, expected_root, [Path("artifact.json")])

    def test_demo_contract_matches_checked_in_goldens(self) -> None:
        from policy_eval_harness.demo import run_demo_from_manifest

        out_dir = self.root / "demo"
        artifacts = run_demo_from_manifest(self.repo_root / "examples" / "core_demo" / "demo.yaml", out_dir)
        golden_root = self.repo_root / "examples" / "core_demo" / "golden"
        assert_artifact_bundle_matches(
            self,
            artifacts.replay_root,
            golden_root / "replay",
            DEMO_REPLAY_RELATIVE_ARTIFACTS,
        )
        assert_artifact_bundle_matches(
            self,
            artifacts.evaluation_root,
            golden_root / "evaluate",
            DEMO_EVALUATE_RELATIVE_ARTIFACTS,
        )

    def test_label_compare_contract_matches_checked_in_goldens(self) -> None:
        from policy_eval_harness.workflows import run_label_compare_from_manifest

        out_dir = self.root / "label-compare"
        artifacts = run_label_compare_from_manifest(
            self.repo_root / "examples" / "label_compare" / "label_compare.yaml",
            out_dir,
        )
        golden_root = self.repo_root / "examples" / "label_compare" / "golden"
        assert_artifact_bundle_matches(
            self,
            artifacts.oos_auc_path.parent,
            golden_root,
            LABEL_COMPARE_RELATIVE_ARTIFACTS,
        )

    def test_ablation_contract_matches_checked_in_goldens(self) -> None:
        from policy_eval_harness.workflows import run_ablation_2x2_from_manifest

        out_dir = self.root / "ablation-2x2"
        artifacts = run_ablation_2x2_from_manifest(
            self.repo_root / "examples" / "ablation_2x2" / "ablation_2x2.yaml",
            out_dir,
        )
        golden_root = self.repo_root / "examples" / "ablation_2x2" / "golden"
        assert_artifact_bundle_matches(
            self,
            artifacts.factor_effects_path.parent,
            golden_root,
            ABLATION_RELATIVE_ARTIFACTS,
        )
