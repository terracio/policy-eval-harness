from __future__ import annotations

from pathlib import Path

import typer

from policy_eval_harness.replay import run_replay_from_manifest

NOT_IMPLEMENTED_MESSAGE = "Not yet implemented in this skeleton."

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


def _not_implemented(manifest: Path, out_dir: Path) -> None:
    del manifest, out_dir
    typer.echo(NOT_IMPLEMENTED_MESSAGE, err=True)
    raise typer.Exit(code=1)


@demo_app.command("run")
def demo_run(
    manifest: Path = typer.Option(..., exists=False, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Run the end-to-end demo workflow."""
    _not_implemented(manifest, out_dir)


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
    manifest: Path = typer.Option(..., exists=False, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., file_okay=False, dir_okay=True),
) -> None:
    """Run an evaluation workflow from a manifest."""
    _not_implemented(manifest, out_dir)


app.add_typer(demo_app, name="demo")
app.add_typer(replay_app, name="replay")
app.add_typer(evaluate_app, name="evaluate")
