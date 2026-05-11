from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer

from policy_eval_harness.demo import run_demo_from_manifest
from policy_eval_harness.evaluation import analyze_policy_iteration_trend, run_evaluation_from_manifest
from policy_eval_harness.replay import canonical_json, run_replay_from_manifest
from policy_eval_harness.workflows import (
    run_ablation_2x2_from_manifest,
    run_label_compare_from_manifest,
)

app = typer.Typer(
    help="CLI for the policy evaluation harness.",
    no_args_is_help=True,
)
demo_app = typer.Typer(
    help="Commands for the bundled demo workflow.",
    no_args_is_help=True,
)
replay_app = typer.Typer(
    help="Commands for deterministic replay workflows.",
    no_args_is_help=True,
)
evaluate_app = typer.Typer(
    help="Commands for evaluation and promotion workflows.",
    no_args_is_help=True,
)
workflow_app = typer.Typer(
    help="Commands for secondary workflow examples.",
    no_args_is_help=True,
)
label_compare_app = typer.Typer(
    help="Commands for label comparison workflows.",
    no_args_is_help=True,
)
ablation_app = typer.Typer(
    help="Commands for 2x2 ablation workflows.",
    no_args_is_help=True,
)


@demo_app.command("run")
def demo_run(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Run the end-to-end demo workflow."""
    artifacts = run_demo_from_manifest(manifest, out_dir)
    typer.echo(str(artifacts.replay_root))
    typer.echo(str(artifacts.evaluation_artifacts.comparison_panel_path))
    typer.echo(str(artifacts.evaluation_artifacts.scorecard_path))
    typer.echo(str(artifacts.evaluation_artifacts.promotion_decisions_path))


@replay_app.command("run")
def replay_run(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Run a replay workflow from a manifest."""
    artifacts = run_replay_from_manifest(manifest, out_dir)
    typer.echo(str(artifacts.episode_summary_path))
    typer.echo(str(artifacts.step_trace_path))
    typer.echo(str(artifacts.run_metadata_path))


@evaluate_app.command("run")
def evaluate_run(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Run an evaluation workflow from a manifest."""
    artifacts = run_evaluation_from_manifest(manifest, out_dir)
    typer.echo(str(artifacts.comparison_panel_path))
    typer.echo(str(artifacts.scorecard_path))
    typer.echo(str(artifacts.promotion_decisions_path))


@evaluate_app.command("trend")
def evaluate_trend(
    run_dirs: list[Path] = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    candidate_variant_id: str | None = typer.Option(None, "--candidate-variant-id"),
    window: int = typer.Option(10, "--window"),
    split: str = typer.Option("holdout", "--split"),
    json_out: Path | None = typer.Option(None, "--json-out", file_okay=True, dir_okay=False),
    exit_on_regression: bool = typer.Option(False, "--exit-on-regression"),
) -> None:
    """Analyze holdout mean_utility trend across evaluation output directories."""
    try:
        report = analyze_policy_iteration_trend(
            run_dirs,
            candidate_variant_id=candidate_variant_id,
            window=window,
            split=split,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Candidate: {report.candidate_variant_id or 'n/a'}")
    typer.echo(f"Split: {report.split}")
    typer.echo(f"Iterations analyzed: {report.iterations_analyzed}")
    typer.echo(
        "mean_utility slope: "
        f"{report.mean_utility_trend.slope:.4f} ({report.mean_utility_trend.direction})"
    )
    typer.echo(f"Promotion rate: {report.promotion_rate:.2%}")
    typer.echo(
        "Verdicts: "
        f"pass={report.pass_count} fail={report.fail_count} no_verdict={report.no_verdict_count}"
    )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(canonical_json(asdict(report)) + "\n", encoding="utf-8")

    if exit_on_regression and report.any_regression:
        raise typer.Exit(code=1)


@label_compare_app.command("run")
def label_compare_run(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Run a label comparison workflow from a manifest."""
    artifacts = run_label_compare_from_manifest(manifest, out_dir)
    typer.echo(str(artifacts.oos_auc_path))
    typer.echo(str(artifacts.summary_path))
    typer.echo(str(artifacts.feature_manifest_snapshot_path))


@ablation_app.command("run")
def ablation_run(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Run a 2x2 ablation workflow from a manifest."""
    artifacts = run_ablation_2x2_from_manifest(manifest, out_dir)
    typer.echo(str(artifacts.factor_effects_path))
    typer.echo(str(artifacts.interaction_summary_path))
    typer.echo(str(artifacts.report_path))


workflow_app.add_typer(label_compare_app, name="label-compare")
workflow_app.add_typer(ablation_app, name="ablation-2x2")
app.add_typer(demo_app, name="demo")
app.add_typer(replay_app, name="replay")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(workflow_app, name="workflow")


if __name__ == "__main__":
    app()
