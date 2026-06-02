import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.classifier import TextClassifier
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.data_models import TextCategory


class TestTextClassifier:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(
            categories=["porn", "violence", "political", "ad", "copyright", "normal"],
            classifier_threshold=0.5,
        )

    @pytest.mark.asyncio
    async def test_classify_returns_category(self, config):
        """Mock 模型返回 normal 类别"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        import torch
        logits = torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0, 5.0]])
        mock_model.return_value.logits = logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            classifier = TextClassifier(config)
            result = await classifier.classify("正常文本")
            assert result.category == TextCategory.NORMAL
            assert result.confidence > 0.5
            assert len(result.scores) == 6

    @pytest.mark.asyncio
    async def test_classify_porn(self, config):
        """分类为 porn"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        import torch
        logits = torch.tensor([[5.0, -2.0, -2.0, -2.0, -2.0, -2.0]])
        mock_model.return_value.logits = logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            classifier = TextClassifier(config)
            result = await classifier.classify("色情内容")
            assert result.category == TextCategory.PORN

    @pytest.mark.asyncio
    async def test_classify_empty_text(self, config):
        """空文本返回 normal"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        import torch
        logits = torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0, 5.0]])
        mock_model.return_value.logits = logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            classifier = TextClassifier(config)
            result = await classifier.classify("")
            assert result.category == TextCategory.NORMAL
