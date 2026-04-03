# Core Demo

This example simulates a sequential approval workflow where a policy can `APPROVE`, `REJECT`, `ESCALATE`, or wait for more evidence.

- `approval_baseline_v1` is the conservative baseline.
- `approval_targeted_v2` is designed to promote on holdout.
- `approval_overactive_v1` is intentionally too eager and should fail promotion.

The main repo docs are added in `CODS-35`; this folder only holds the runnable public-safe demo assets.

Checked-in golden outputs live under `golden/`. `demo.yaml` regenerates the full bundle, while `evaluate.yaml`
can also be run on its own against the checked-in replay goldens.
