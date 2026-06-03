import logging

from src.illegal_review.data_models import (
    PreprocessingResult, TextAnalysisResult,
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
        prep_result: PreprocessingResult,
    ) -> TextAnalysisResult:
        """完整分析：从预处理结果中提取 OCR + 语音来源"""
        tasks = {}

        if prep_result.ocr_results:
            ocr_text = TextAdaptor.extract_ocr_text(
                prep_result.ocr_results,
                min_confidence=self._config.ocr_confidence_threshold,
                max_length=self._config.ocr_max_text_length,
            )
            if ocr_text:
                tasks["ocr"] = self._analyzer.analyze(ocr_text, source="ocr")

        if prep_result.transcript:
            trans_text = TextAdaptor.extract_transcript(
                prep_result.transcript,
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
            video_id=prep_result.input_id,
        )
