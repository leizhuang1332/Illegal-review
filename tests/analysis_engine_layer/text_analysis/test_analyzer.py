import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.analyzer import TextAnalyzer
from src.illegal_review.config.settings import TextAnalysisConfig


class TestTextAnalyzer:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(
            sensitive_fuzzy_match_enabled=False,
            sensitive_word_list_path="",
            ner_enabled=False,
        )

    @pytest.fixture
    def analyzer(self, config):
        # Configure mock instances so async methods return AsyncMock (awaitable)
        sem_inst = MagicMock()
        sem_inst.encode = AsyncMock(return_value=None)

        sens_inst = MagicMock()
        sens_inst.match_all = AsyncMock(return_value=[])

        sent_inst = MagicMock()
        sent_inst.analyze = AsyncMock(return_value=0.0)

        ner_inst = MagicMock()
        ner_inst.recognize = AsyncMock(return_value=[])

        cls_inst = MagicMock()
        cls_inst.classify = AsyncMock(return_value=None)

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.SemanticEncoder", return_value=sem_inst), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.SensitiveMatcher", return_value=sens_inst), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.SentimentAnalyzer", return_value=sent_inst), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.NERecognizer", return_value=ner_inst), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.TextClassifier", return_value=cls_inst):
            return TextAnalyzer(config)

    @pytest.mark.asyncio
    async def test_analyze_returns_source_analysis(self, analyzer):
        result = await analyzer.analyze("测试文本", source="ocr")
        assert result.source == "ocr"
        assert result.text_length == 4

    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, analyzer):
        result = await analyzer.analyze("", source="ocr")
        assert result is not None
        assert result.text_length == 0

    @pytest.mark.asyncio
    async def test_analyze_multiple_sources(self, analyzer):
        ocr_result = await analyzer.analyze("OCR文本", source="ocr")
        transcript_result = await analyzer.analyze("语音文本", source="transcript")
        assert ocr_result.source == "ocr"
        assert transcript_result.source == "transcript"
