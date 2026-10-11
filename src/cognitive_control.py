"""
Cognitive Uncertainty Auditor & Calibrated Decision Control.

Evaluates match candidates, score distributions, and margins to detect:
1. High-confidence match (ACCEPT)
2. Low evidence / Out-of-domain query (BELİRSİZ_VEYA_BİLİNMİYOR)
3. Ambiguous or conflicting candidates (ÇELİŞKİLİ_KANIT)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class DecisionType(Enum):
    ACCEPT = "ACCEPT"
    BELIRSIZ_VEYA_BILINMIYOR = "BELİRSİZ_VEYA_BİLİNMİYOR"
    CELISKILI_KANIT = "ÇELİŞKİLİ_KANIT"


@dataclass
class CognitiveDecision:
    decision_type: DecisionType
    top_block_id: Optional[str]
    confidence_score: float
    margin: float
    audit_explanation: str


class CognitiveUncertaintyAuditor:
    """
    Audits block evaluation results and enforces explicit cognitive decision policies.
    """
    def __init__(self, accept_threshold: float = 0.55, margin_threshold: float = 0.05):
        self.accept_threshold = accept_threshold
        self.margin_threshold = margin_threshold

    def audit(
        self,
        candidate_scores: List[Tuple[str, float]],
        query_text: str = ""
    ) -> CognitiveDecision:
        """
        Audit a ranked list of (block_id, similarity_score) pairs.
        """
        if not candidate_scores:
            return CognitiveDecision(
                decision_type=DecisionType.BELIRSIZ_VEYA_BILINMIYOR,
                top_block_id=None,
                confidence_score=0.0,
                margin=0.0,
                audit_explanation="Hiçbir blok adayı bulunamadı."
            )

        # Sort by similarity descending
        sorted_candidates = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        top1_id, top1_score = sorted_candidates[0]

        if len(sorted_candidates) == 1:
            margin = top1_score
            top2_score = 0.0
        else:
            top2_id, top2_score = sorted_candidates[1]
            margin = top1_score - top2_score

        # Rule 1: Below absolute minimum accept threshold -> BELİRSİZ_VEYA_BİLİNMİYOR
        if top1_score < self.accept_threshold:
            return CognitiveDecision(
                decision_type=DecisionType.BELIRSIZ_VEYA_BILINMIYOR,
                top_block_id=top1_id,
                confidence_score=top1_score,
                margin=margin,
                audit_explanation=(
                    f"Top-1 skoru ({top1_score:.4f}) kabul eşiğinin ({self.accept_threshold:.4f}) altında. "
                    f"Sorgu bilinmeyen/yetersiz bilgi içeriyor."
                )
            )

        # Rule 2: Top-1 score is high, but margin with Top-2 is too narrow -> ÇELİŞKİLİ_KANIT
        if len(sorted_candidates) > 1 and margin < self.margin_threshold:
            return CognitiveDecision(
                decision_type=DecisionType.CELISKILI_KANIT,
                top_block_id=top1_id,
                confidence_score=top1_score,
                margin=margin,
                audit_explanation=(
                    f"En yüksek iki blok ({top1_id} vs {sorted_candidates[1][0]}) skorları çok yakın "
                    f"({top1_score:.4f} vs {top2_score:.4f}, fark={margin:.4f} < {self.margin_threshold:.4f}). "
                    f"Çelişkili kanıt veya yüksek belirsizlik tespit edildi."
                )
            )

        # Rule 3: Single confident block -> ACCEPT
        return CognitiveDecision(
            decision_type=DecisionType.ACCEPT,
            top_block_id=top1_id,
            confidence_score=top1_score,
            margin=margin,
            audit_explanation=f"Top-1 blok ({top1_id}) yüksek güvenilirlikle ({top1_score:.4f}) kabul edildi."
        )
