from __future__ import annotations

from typing import Mapping, Optional

from policy_eval_harness.replay import ReplayStepInput, ReplayTransition


def threshold_policy(
    step_input: ReplayStepInput,
    params: Optional[Mapping[str, object]] = None,
) -> Optional[dict]:
    params = params or {}
    if step_input.case.case_id == params.get("raise_on_case"):
        raise RuntimeError("policy boom")
    threshold = params.get("threshold", 0)
    score = step_input.step.observation.get("score", 0)
    if score >= threshold:
        return {
            "decision": "act",
            "score": score,
            "variant": step_input.variant_id,
        }
    return None


def replay_executor(
    step_input: ReplayStepInput,
    action: Optional[dict],
    params: Optional[Mapping[str, object]] = None,
) -> ReplayTransition:
    params = params or {}
    if step_input.case.case_id == params.get("raise_on_case"):
        raise RuntimeError("executor boom")

    current_state = dict(step_input.state or {})
    history = list(current_state.get("history", []))
    history.append(
        {
            "step_index": step_input.step.step_index,
            "action": action,
        }
    )
    terminated = bool(step_input.step.observation.get("stop"))
    return ReplayTransition(
        next_state={"history": history},
        terminated=terminated,
        termination_reason="stop_flag" if terminated else None,
        final_status="terminated" if terminated else "completed",
        metrics={
            "actions": 1 if action is not None else 0,
            "steps": 1,
        },
        trace={
            "stop": terminated,
            "variant": step_input.variant_id,
        },
    )
