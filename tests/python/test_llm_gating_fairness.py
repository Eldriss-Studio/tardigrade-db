"""ATDD: fairness preservation under Decorator widening.

The Decorator's private inner retrieval budget must not surface in
``validate_fairness``: every system still presents the same
profile-level ``top_k``, ``answerer_model``, ``judge_model``, and
prompts to the fairness validator.
"""

from __future__ import annotations

import pytest

from tdb_bench.fairness import FairnessError, validate_fairness


_PROFILE_TOP_K = 5
_ANSWERER = "deepseek-chat"
_JUDGE = "deepseek-chat"
_ANSWER_PROMPT = "Answer concisely using retrieved evidence."
_JUDGE_PROMPT = "Score factual correctness and evidence relevance only."


def _shared_cfg(**override) -> dict:
    return {
        "top_k": _PROFILE_TOP_K,
        "answerer_model": _ANSWERER,
        "judge_model": _JUDGE,
        "answer_prompt": _ANSWER_PROMPT,
        "judge_prompt": _JUDGE_PROMPT,
        **override,
    }


class TestFairnessPreservedAcrossSystems:
    def test_baseline_and_gated_systems_both_pass_with_same_top_k(self):
        # Both systems present the profile's top_k=5 to the validator.
        # The Decorator's private inner top_k=25 lives below this line
        # and is not part of the fairness contract.
        validate_fairness({
            "tardigrade": _shared_cfg(),
            "tardigrade-llm-gated": _shared_cfg(),
        })

    def test_differing_top_k_raises(self):
        # Sanity: the validator still flags real mismatches, so the
        # contract above isn't accidentally permissive.
        with pytest.raises(FairnessError, match="FAIRNESS_MISMATCH"):
            validate_fairness({
                "tardigrade": _shared_cfg(),
                "tardigrade-llm-gated": _shared_cfg(top_k=25),
            })

    def test_three_systems_share_baseline(self):
        validate_fairness({
            "tardigrade": _shared_cfg(),
            "tardigrade-llm-gated": _shared_cfg(),
            "mem0_oss": _shared_cfg(),
        })
