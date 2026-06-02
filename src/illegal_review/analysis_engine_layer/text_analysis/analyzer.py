import asyncio
import logging
from typing import Any, List

from src.illegal_review.data_models import SourceAnalysis
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.analysis_engine_layer.text_analysis.preprocessor import TextPreprocessor
from src.illegal_review.analysis_engine_layer.text_analysis.semantic import SemanticEncoder
from src.illegal_review.analysis_engine_layer.text_analysis.sensitive_matcher import SensitiveMatcher
from src.illegal_review.analysis_engine_layer.text_analysis.sentiment import SentimentAnalyzer
from src.illegal_review.analysis_engine_layer.text_analysis.ner import NERecognizer
from src.illegal_review.analysis_engine_layer.text_analysis.classifier import TextClassifier

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """文本分析器 — 串联所有模块，并行执行互不依赖的分析"""

    def __init__(self, config: TextAnalysisConfig):
        self._preprocessor = TextPreprocessor()
        self._semantic = SemanticEncoder(config)
        self._sensitive = SensitiveMatcher(config)
        self._sentiment = SentimentAnalyzer(config)
        self._ner = NERecognizer(config)
        self._classifier = TextClassifier(config)

    async def analyze(self, text: str, source: str) -> SourceAnalysis:
        """完整分析流水线：预处理 → 5模块并行 → SourceAnalysis"""
        cleaned = self._preprocessor.process(text)

        results = await asyncio.gather(
            self._semantic.encode(cleaned.text),
            self._sensitive.match_all(cleaned.text),
            self._sentiment.analyze(cleaned.text),
            self._ner.recognize(cleaned.text),
            self._classifier.classify(cleaned.text),
            return_exceptions=True,
        )

        embed_result, sensitive_result, sentiment_result, ner_result, classify_result = results
        errors: List[str] = []

        embedding = self._unwrap(embed_result, errors, "semantic_encoding")
        sensitive_words = self._unwrap(sensitive_result, errors, "sensitive_matching") or []
        sentiment_score = self._unwrap(sentiment_result, errors, "sentiment_analysis")
        entities = self._unwrap(ner_result, errors, "ner") or []
        category = self._unwrap(classify_result, errors, "classification")

        return SourceAnalysis(
            source=source,
            text_length=len(text),
            semantic_embedding=embedding,
            sensitive_words=sensitive_words,
            sentiment_score=sentiment_score,
            entities=entities,
            category=category,
            errors=errors,
        )

    def _unwrap(self, result: Any, errors: List[str], module: str) -> Any:
        """异常 → None + 记录错误"""
        if isinstance(result, Exception):
            errors.append(f"{module}: {result}")
            return None
        return result
