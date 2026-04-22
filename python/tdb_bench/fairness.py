"""Fairness guards for cross-system comparability."""

from __future__ import annotations


class FairnessError(ValueError):
    """Raised when systems are not comparable under shared evaluator settings."""


def validate_fairness(system_cfg: dict[str, dict]) -> None:
    """Require identical top_k/models/prompts across systems."""
    if not system_cfg:
        raise FairnessError("No systems configured")

    keys = ["top_k", "answerer_model", "judge_model", "answer_prompt", "judge_prompt"]
    first_name = next(iter(system_cfg.keys()))
    baseline = {k: system_cfg[first_name].get(k) for k in keys}

    for name, cfg in system_cfg.items():
        current = {k: cfg.get(k) for k in keys}
        if current != baseline:
            raise FairnessError(
                f"FAIRNESS_MISMATCH: system={name} current={current} baseline={baseline}"
            )
