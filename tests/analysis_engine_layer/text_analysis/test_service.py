import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.service import TextAnalysisService
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.data_models import OCRResult, TextAnalysisResult


class TestTextAnalysisService:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(
            sensitive_fuzzy_match_enabled=False,
            sensitive_word_list_path="",
        )

    @pytest.fixture
    def service(self, config):
        with patch("src.illegal_review.analysis_engine_layer.text_analysis.service.TextAnalyzer"), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.service.ResultMerger"):
            return TextAnalysisService(config)

    @pytest.mark.asyncio
    async def test_analyze_all_with_ocr_and_transcript(self, service):
        """OCR + 语音都有的正常流程"""
        video_id = uuid4()
        ocr_results = [
            OCRResult(text="测试文字", confidence=0.9, bbox=None, frame_index=0),
        ]
        transcript = "这是一段语音转写文本"

        from src.illegal_review.data_models import SourceAnalysis
        mock_result = SourceAnalysis(source="ocr", text_length=4)
        service._analyzer.analyze = AsyncMock(return_value=mock_result)

        service._merger.merge = MagicMock(return_value=TextAnalysisResult(
            video_id=video_id,
            ocr=mock_result,
        ))

        result = await service.analyze_all(video_id, ocr_results, transcript)
        assert isinstance(result, TextAnalysisResult)
        assert result.video_id == video_id

    @pytest.mark.asyncio
    async def test_analyze_all_no_inputs(self, service):
        """无 OCR 无语音 → 空结果"""
        video_id = uuid4()
        service._merger.merge = MagicMock(return_value=TextAnalysisResult(
            video_id=video_id,
        ))
        result = await service.analyze_all(video_id, ocr_results=None, transcript=None)
        assert result.video_id == video_id
        assert result.ocr is None
        assert result.transcript is None
