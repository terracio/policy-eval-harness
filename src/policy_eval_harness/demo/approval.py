from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from policy_eval_harness.replay import ReplayStepInput, ReplayTransition


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _current_state(step_input: ReplayStepInput) -> Dict[str, float]:
    previous = step_input.state if isinstance(step_input.state, Mapping) else {}
    observation = step_input.step.observation
    return {
        "max_approve": max(_as_float(previous.get("max_approve")), _as_float(observation.get("approve_signal"))),
        "max_reject": max(_as_float(previous.get("max_reject")), _as_float(observation.get("reject_signal"))),
        "max_escalate": max(_as_float(previous.get("max_escalate")), _as_float(observation.get("escalate_signal"))),
        "steps_seen": int(previous.get("steps_seen", 0)) + 1,
    }


def _decision(decision: str, state: Mapping[str, float], reason: str) -> Dict[str, Any]:
    return {
        "decision": decision,
        "reason": reason,
        "max_approve": state["max_approve"],
        "max_reject": state["max_reject"],
        "max_escalate": state["max_escalate"],
        "steps_seen": state["steps_seen"],
    }


def approval_baseline_v1(
    step_input: ReplayStepInput,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[Mapping[str, Any]]:
    del params
    state = _current_state(step_input)
    margin = state["max_approve"] - state["max_reject"]

    if state["max_escalate"] >= 0.82 and state["steps_seen"] >= 3:
        return _decision("ESCALATE", state, "strong_escalation_signal")
    if state["max_approve"] >= 0.88 and margin >= 0.25:
        return _decision("APPROVE", state, "high_approve_confidence")
    if state["max_reject"] >= 0.88 and margin <= -0.25:
        return _decision("REJECT", state, "high_reject_confidence")
    if state["steps_seen"] >= 5:
        if margin >= 0.18:
            return _decision("APPROVE", state, "late_majority_approve")
        if margin <= -0.18:
            return _decision("REJECT", state, "late_majority_reject")
        if state["max_escalate"] >= 0.66:
            return _decision("ESCALATE", state, "late_escalation_backstop")
    return None


def approval_targeted_v2(
    step_input: ReplayStepInput,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[Mapping[str, Any]]:
    del params
    state = _current_state(step_input)
    margin = state["max_approve"] - state["max_reject"]

    if state["max_escalate"] >= 0.72 and state["steps_seen"] >= 2:
        return _decision("ESCALATE", state, "targeted_escalation")
    if state["max_approve"] >= 0.80 and margin >= 0.18:
        return _decision("APPROVE", state, "earlier_approve_confidence")
    if state["max_reject"] >= 0.80 and margin <= -0.18:
        return _decision("REJECT", state, "earlier_reject_confidence")
    if state["steps_seen"] >= 4:
        if state["max_escalate"] >= 0.62:
            return _decision("ESCALATE", state, "ambiguous_case_escalation")
        if margin >= 0.12:
            return _decision("APPROVE", state, "late_targeted_approve")
        if margin <= -0.12:
            return _decision("REJECT", state, "late_targeted_reject")
    return None


def approval_overactive_v1(
    step_input: ReplayStepInput,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[Mapping[str, Any]]:
    del params
    state = _current_state(step_input)
    margin = abs(state["max_approve"] - state["max_reject"])

    if state["max_escalate"] >= 0.45:
        return _decision("ESCALATE", state, "overactive_escalation")
    if state["steps_seen"] >= 2 and margin <= 0.12:
        return _decision("ESCALATE", state, "low_margin_escalation")
    if state["max_approve"] >= 0.72:
        return _decision("APPROVE", state, "early_approve")
    if state["max_reject"] >= 0.72:
        return _decision("REJECT", state, "early_reject")
    return None


def approval_workflow_executor(
    step_input: ReplayStepInput,
    action: Optional[Mapping[str, Any]],
    params: Optional[Mapping[str, Any]] = None,
) -> ReplayTransition:
    del params
    case = step_input.case
    state = _current_state(step_input)
    metadata = case.metadata
    max_steps = int(metadata.get("max_steps", 6))
    decision = None if action is None else str(action.get("decision", "")).upper()

    if not decision:
        if step_input.step.step_index + 1 >= max_steps:
            return ReplayTransition(
                next_state=state,
                terminated=True,
                termination_reason="max_steps",
                final_status="completed",
                metrics={
                    "utility": float(metadata.get("max_steps_penalty", -0.9)),
                    "timed_out_cases": 1.0,
                },
                trace={
                    "outcome": "max_steps",
                    "truth_action": metadata.get("truth_action"),
                },
            )
        return ReplayTransition(
            next_state=state,
            terminated=False,
            final_status="in_progress",
            metrics={"utility": -float(metadata.get("delay_penalty", 0.06))},
            trace={"outcome": "wait"},
        )

    truth_action = str(metadata.get("truth_action", "")).upper()
    correct_reward = float(metadata.get("correct_reward", 1.0))
    wrong_decision_penalty = float(metadata.get("wrong_decision_penalty", -1.1))
    escalate_reward = float(metadata.get("escalate_reward", 0.7))
    easy_escalate_penalty = float(metadata.get("easy_escalate_penalty", -0.35))

    if decision == "ESCALATE":
        utility = escalate_reward if truth_action == "ESCALATE" else easy_escalate_penalty
        outcome = "correct_escalation" if truth_action == "ESCALATE" else "unnecessary_escalation"
    elif decision == truth_action:
        utility = correct_reward
        outcome = "correct_terminal_decision"
    else:
        utility = wrong_decision_penalty
        outcome = "incorrect_terminal_decision"

    return ReplayTransition(
        next_state=state,
        terminated=True,
        termination_reason=decision.lower(),
        final_status="completed",
        metrics={"utility": utility},
        trace={
            "decision": decision,
            "outcome": outcome,
            "policy_reason": None if action is None else action.get("reason"),
            "truth_action": truth_action,
        },
    )
