# Design Principles

## Determinism Over Convenience

If a result cannot be reproduced from checked-in inputs, it is weak evidence. This repo prefers fixed universes, stable ordering, and inspectable artifacts over more convenient but opaque workflows.

## Paired Comparisons Over Vibes

A candidate should not win because it saw an easier slice of the world. Baseline and candidate must face the same cases, and missing paired coverage should be treated as a first-class diagnostic.

## Explicit Gates Over Preference Drift

Research loops degrade when “better” is left implicit. Promotion should depend on declared thresholds and an authoritative split, not on whichever run felt most convincing in the moment.

## Utility Plus Diagnostics

Utility is the decision target, but it is not enough by itself. A useful evaluation workflow also reports coverage, pathologies, and execution failures so regressions are visible before promotion.

## Complement Platforms, Do Not Replace Them

Tracing systems, eval dashboards, and rubric-based scoring can all add value. This repo focuses on the narrower problem of deterministic replay and promotion-gated iteration for sequential policies.

## Local-First, Inspectable Artifacts

The public demo should run without private infrastructure. CSV, Parquet, YAML, JSON, and Markdown are intentional choices: they make the workflow portable and auditable.

## Research-Informed, Not Theory-Maximalist

The contribution here is not a new theory of evaluation. It is the applied synthesis: taking ideas from sequential decision-making, ablation discipline, and offline evaluation, then turning them into a workflow practitioners can actually run.
