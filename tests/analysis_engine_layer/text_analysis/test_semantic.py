import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.semantic import SemanticEncoder
from src.illegal_review.config.settings import TextAnalysisConfig


class TestSemanticEncoder:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(bert_model="bert-base-chinese")

    @pytest.fixture
    def mock_model(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        import torch
        mock_output.last_hidden_state[:, 0, :].squeeze.return_value = torch.zeros(768)
        mock_model.return_value = mock_output

        return mock_tokenizer, mock_model

    @pytest.mark.asyncio
    async def test_encode_returns_768d_vector(self, config, mock_model):
        mock_tokenizer, mock_model = mock_model
        with patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoModel.from_pretrained", return_value=mock_model):
            encoder = SemanticEncoder(config)
            embedding = await encoder.encode("测试文本")
            assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_encode_empty_text(self, config, mock_model):
        mock_tokenizer, mock_model = mock_model
        with patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoModel.from_pretrained", return_value=mock_model):
            encoder = SemanticEncoder(config)
            embedding = await encoder.encode("")
            assert len(embedding) == 768
