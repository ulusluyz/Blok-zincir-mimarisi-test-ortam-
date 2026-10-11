"""
Unit tests for Cognitive Uncertainty Auditor & Calibrated Decision Control.
"""

import pytest
from src.cognitive_control import (
    CognitiveUncertaintyAuditor, DecisionType
)


def test_cognitive_auditor_accept():
    auditor = CognitiveUncertaintyAuditor(accept_threshold=0.55, margin_threshold=0.05)
    candidates = [("Block_Selam", 0.85), ("Block_Matematik", 0.40)]
    decision = auditor.audit(candidates)

    assert decision.decision_type == DecisionType.ACCEPT
    assert decision.top_block_id == "Block_Selam"
    assert decision.confidence_score == 0.85


def test_cognitive_auditor_unknown():
    auditor = CognitiveUncertaintyAuditor(accept_threshold=0.55, margin_threshold=0.05)
    candidates = [("Block_Selam", 0.45), ("Block_Matematik", 0.42)]
    decision = auditor.audit(candidates)

    assert decision.decision_type == DecisionType.BELIRSIZ_VEYA_BILINMIYOR
    assert decision.top_block_id == "Block_Selam"


def test_cognitive_auditor_conflicting_evidence():
    auditor = CognitiveUncertaintyAuditor(accept_threshold=0.55, margin_threshold=0.05)
    candidates = [("Block_Fizik", 0.78), ("Block_Matematik", 0.76)]
    decision = auditor.audit(candidates)

    assert decision.decision_type == DecisionType.CELISKILI_KANIT
    assert decision.top_block_id == "Block_Fizik"
    assert decision.margin < 0.05
