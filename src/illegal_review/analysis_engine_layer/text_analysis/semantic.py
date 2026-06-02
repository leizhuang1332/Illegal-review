import asyncio
import logging
from typing import List

from transformers import AutoTokenizer, AutoModel

from src.illegal_review.config.settings import TextAnalysisConfig

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """BERT 语义编码器 — 文本 → 768维语义向量"""

    def __init__(self, config: TextAnalysisConfig):
        logger.info(f"Loading BERT model: {config.bert_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self._model = AutoModel.from_pretrained(config.bert_model)
        self._model.eval()
        logger.info(f"BERT model '{config.bert_model}' loaded")

    async def encode(self, text: str) -> List[float]:
        """文本 → 768维语义向量（线程池推理）"""
        return await asyncio.to_thread(self._encode_sync, text)

    def _encode_sync(self, text: str) -> List[float]:
        import torch
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        if isinstance(embedding, float):
            embedding = [embedding]
        return embedding
