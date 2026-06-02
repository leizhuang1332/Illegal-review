import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.ner import NERecognizer
from src.illegal_review.config.settings import TextAnalysisConfig


class TestNERecognizer:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(ner_enabled=True)

    @pytest.mark.asyncio
    async def test_recognize_returns_entities(self, config):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "张", "三", "去", "北", "京", "[SEP]"]

        mock_model = MagicMock()
        import torch
        mock_model.config.id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
        mock_logits = torch.zeros(1, 7, 5)
        mock_logits[0, 1, 1] = 10  # "张" → B-PER
        mock_logits[0, 2, 2] = 10  # "三" → I-PER
        mock_logits[0, 4, 3] = 10  # "北" → B-LOC
        mock_logits[0, 5, 4] = 10  # "京" → I-LOC
        mock_model.return_value.logits = mock_logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoModelForTokenClassification.from_pretrained", return_value=mock_model):
            recognizer = NERecognizer(config)
            entities = await recognizer.recognize("张三去北京")

            assert len(entities) == 2
            names = {e.name for e in entities}
            assert "张三" in names
            assert "北京" in names

    @pytest.mark.asyncio
    async def test_no_entities(self, config):
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "正", "常", "文", "本", "[SEP]"]

        mock_model = MagicMock()
        import torch
        mock_model.config.id2label = {0: "O"}
        mock_logits = torch.zeros(1, 6, 1)
        mock_model.return_value.logits = mock_logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoModelForTokenClassification.from_pretrained", return_value=mock_model):
            recognizer = NERecognizer(config)
            entities = await recognizer.recognize("正常文本")
            assert len(entities) == 0
