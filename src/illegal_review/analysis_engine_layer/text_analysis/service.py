import logging
from typing import List, Optional
from uuid import UUID

from src.illegal_review.data_models import (
    OCRResult, TextAnalysisResult,
)
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.analysis_engine_layer.text_analysis.analyzer import TextAnalyzer
from src.illegal_review.analysis_engine_layer.text_analysis.adaptor import TextAdaptor
from src.illegal_review.analysis_engine_layer.text_analysis.merger import ResultMerger

logger = logging.getLogger(__name__)


class TextAnalysisService:
    """文本分析引擎对外门面"""

    def __init__(self, config: TextAnalysisConfig):
        self._config = config
        self._analyzer = TextAnalyzer(config)
        self._merger = ResultMerger()

    async def analyze_all(
        self,
        video_id: UUID,
        ocr_results: Optional[List[OCRResult]] = None,
        transcript: Optional[str] = None,
    ) -> TextAnalysisResult:
        """完整分析：OCR + 语音，自动跳过 None 来源"""
        tasks = {}

        if ocr_results:
            ocr_text = TextAdaptor.extract_ocr_text(
                ocr_results,
                min_confidence=self._config.ocr_confidence_threshold,
                max_length=self._config.ocr_max_text_length,
            )
            if ocr_text:
                tasks["ocr"] = self._analyzer.analyze(ocr_text, source="ocr")

        if transcript:
            trans_text = TextAdaptor.extract_transcript(
                transcript,
                max_length=self._config.max_text_length,
            )
            if trans_text:
                tasks["transcript"] = self._analyzer.analyze(trans_text, source="transcript")

        results = {}
        if tasks:
            for name, task in tasks.items():
                results[name] = await task

        return self._merger.merge(
            ocr_result=results.get("ocr"),
            transcript_result=results.get("transcript"),
            video_id=video_id,
        )
