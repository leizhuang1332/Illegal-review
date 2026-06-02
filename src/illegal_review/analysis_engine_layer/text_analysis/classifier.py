import asyncio
import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.illegal_review.data_models import CategoryResult
from src.illegal_review.config.settings import TextAnalysisConfig

logger = logging.getLogger(__name__)


class TextClassifier:
    """文本违规分类器 — bert + 分类头"""

    def __init__(self, config: TextAnalysisConfig):
        self._categories = config.categories
        self._threshold = config.classifier_threshold
        model_name = config.classifier_model_path or config.bert_model

        logger.info(f"Loading text classifier: {model_name} (num_labels={len(self._categories)})")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self._categories)
        )
        self._model.eval()
        logger.info("Text classifier loaded")

    async def classify(self, text: str) -> CategoryResult:
        """文本分类 → CategoryResult（线程池推理）"""
        return await asyncio.to_thread(self._classify_sync, text)

    def _classify_sync(self, text: str) -> CategoryResult:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        scores = {cat: float(probs[i]) for i, cat in enumerate(self._categories)}
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        return CategoryResult(
            category=self._categories[pred_idx],
            confidence=confidence,
            scores=scores,
        )
