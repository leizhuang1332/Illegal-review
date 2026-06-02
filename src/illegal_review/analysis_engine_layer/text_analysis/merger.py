import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from src.illegal_review.data_models import (
    SourceAnalysis, TextAnalysisResult, ViolationDetection,
    TextCategory,
)

logger = logging.getLogger(__name__)


class ResultMerger:
    """合并 OCR 和语音的分析结果"""

    def merge(
        self,
        ocr_result: Optional[SourceAnalysis],
        transcript_result: Optional[SourceAnalysis],
        video_id: UUID,
    ) -> TextAnalysisResult:
        violations: List[ViolationDetection] = []

        for src in [ocr_result, transcript_result]:
            if src is None:
                continue
            for sw in src.sensitive_words:
                violations.append(ViolationDetection(
                    type="sensitive_word",
                    category=sw.category,
                    confidence=1.0,
                    timestamp=None,
                    context=None,
                    evidence={"source": src.source, "word": sw.word},
                ))
            if src.category and src.category.category != TextCategory.NORMAL:
                violations.append(ViolationDetection(
                    type="text_classification",
                    category=src.category.category.value,
                    confidence=src.category.confidence,
                    timestamp=None,
                    context=None,
                    evidence={"source": src.source, "scores": src.category.scores},
                ))

        violations = self._deduplicate(violations)

        return TextAnalysisResult(
            video_id=video_id,
            ocr=ocr_result,
            transcript=transcript_result,
            violations=violations,
            processing_stats={
                "violations_count": float(len(violations)),
                "has_ocr": float(ocr_result is not None),
                "has_transcript": float(transcript_result is not None),
            },
        )

    def _deduplicate(self, violations: List[ViolationDetection]) -> List[ViolationDetection]:
        seen: Dict[Tuple[str, str], ViolationDetection] = {}
        for v in violations:
            key = (v.type, v.category)
            if key not in seen or v.confidence > seen[key].confidence:
                seen[key] = v
        return list(seen.values())
