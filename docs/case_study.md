# Case Study

## Question

Should a targeted review policy be promoted over the current baseline in a sequential approval workflow?

This case study uses only the checked-in public demo under `examples/core_demo/`. No private results are required.

## Demo Setup

The demo simulates a reviewer policy operating over time as evidence arrives for each case.

- action space: `WAIT`, `APPROVE`, `REJECT`, `ESCALATE`
- universe size: `24` cases
- split: `16` dev, `8` holdout
- cohorts: clean approve, clean reject, late-reveal, ambiguous

All three variants replay the exact same ordered universe:

- `approval_baseline_v1`: conservative baseline that waits longer before deciding
- `approval_targeted_v2`: candidate intended to act earlier on easy cases while preserving escalation on ambiguous ones
- `approval_overactive_v1`: intentionally over-eager variant that fires too early

## Evaluation Contract

The demo uses the `replay_outcomes_v1` profile with the `sequential_promotion_v1` gate profile.

Holdout promotion thresholds come from [`examples/core_demo/evaluate.yaml`](../examples/core_demo/evaluate.yaml):

- `min_delta_mean_utility >= 0.08`
- `min_paired_coverage_rate >= 1.0`
- `max_candidate_pathology_rate <= 0.26`
- `max_candidate_error_case_rate <= 0.0`

Promotion is based on holdout only.

## What The Replay Shows

The replay artifacts already reveal the behavioral story before any final verdict.

Baseline behavior:

- clean approve and reject cases usually terminate in `4` steps
- late-reveal and ambiguous cases often take `5` steps
- holdout mean utility is `0.79`

Targeted candidate behavior:

- clean approve and reject cases terminate in `3` steps
- late-reveal cases terminate in `4` steps
- ambiguous cases still escalate instead of forcing unsafe terminal actions
- holdout mean utility rises to `0.885`

Overactive behavior:

- many easy and late-reveal cases terminate in `1` or `2` steps
- that speed is not disciplined; it triggers harmful early escalations
- holdout mean utility collapses to `-0.15`

Those values can be verified directly in the checked-in replay summary at [`examples/core_demo/golden/replay/_combined/episode_summary.csv`](../examples/core_demo/golden/replay/_combined/episode_summary.csv).

## Scorecard

The holdout scorecard in [`examples/core_demo/golden/evaluate/scorecard.csv`](../examples/core_demo/golden/evaluate/scorecard.csv) gives the paired comparison against `approval_baseline_v1`.

| Candidate | Holdout mean utility | Delta vs baseline | Paired coverage | Pathology rate | Error case rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `approval_targeted_v2` | `0.885` | `+0.095` | `1.0` | `0.0` | `0.0` |
| `approval_overactive_v1` | `-0.15` | `-0.94` | `1.0` | `0.0` | `0.0` |

The important point is not just who won. It is why:

- both candidates preserved full paired coverage on holdout
- neither introduced pathology or execution errors
- only the targeted candidate cleared the utility threshold

## Promotion Verdict

The final verdicts live in [`examples/core_demo/golden/evaluate/promotion_decisions.json`](../examples/core_demo/golden/evaluate/promotion_decisions.json).

`approval_targeted_v2` is promoted because it passes every holdout gate:

- `delta_mean_utility = +0.095`, above the `0.08` threshold
- `paired_coverage_rate = 1.0`
- `candidate_pathology_rate = 0.0`
- `candidate_error_case_rate = 0.0`

`approval_overactive_v1` fails because its utility regresses badly:

- `delta_mean_utility = -0.94`
- the reliability gates pass, but promotion still fails because the primary utility gate does not

This is the methodological point of the repo. The targeted policy is not promoted because it “looks smarter.” It is promoted because it clears explicit gates on the authoritative split under paired comparison.

## Why This Matters

This case is small by design, but it demonstrates the real workflow:

1. Freeze the evaluation universe.
2. Replay baseline and candidate policies under identical conditions.
3. Measure utility and guardrails on paired holdout data.
4. Promote only if the gates pass.

That is the difference between qualitative policy iteration and decision-grade policy iteration.
