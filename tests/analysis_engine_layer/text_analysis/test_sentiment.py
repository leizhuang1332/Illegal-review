import pytest
from src.illegal_review.analysis_engine_layer.text_analysis.sentiment import SentimentAnalyzer
from src.illegal_review.config.settings import TextAnalysisConfig


class TestSentimentAnalyzer:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig()

    @pytest.fixture
    def analyzer(self, config):
        return SentimentAnalyzer(config)

    def test_positive_text(self, analyzer):
        score = analyzer.analyze("今天天气真好，心情非常愉快")
        assert score > 0

    def test_negative_text(self, analyzer):
        score = analyzer.analyze("太糟糕了，真是令人愤怒")
        assert score < 0

    def test_neutral_text(self, analyzer):
        score = analyzer.analyze("今天星期二")
        assert -0.3 <= score <= 0.3

    def test_negation_reversal(self, analyzer):
        """否定词反转情感极性"""
        negative = analyzer.analyze("不好")
        positive = analyzer.analyze("好")
        assert negative < positive

    def test_adverb_weight(self, analyzer):
        """程度副词增强情感强度"""
        strong = analyzer.analyze("非常喜欢")
        weak = analyzer.analyze("喜欢")
        assert strong > weak

    def test_empty_text(self, analyzer):
        score = analyzer.analyze("")
        assert score == 0.0
