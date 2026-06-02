import pytest
import tempfile
from pathlib import Path
from src.illegal_review.analysis_engine_layer.text_analysis.sensitive_matcher import SensitiveMatcher
from src.illegal_review.config.settings import TextAnalysisConfig


@pytest.fixture
def word_list_path():
    """创建临时敏感词表"""
    content = [
        "毒品,illegal_drugs",
        "赌博,gambling",
        "暴力,violence",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(content))
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def config(word_list_path):
    return TextAnalysisConfig(
        sensitive_word_list_path=word_list_path,
        sensitive_fuzzy_match_enabled=False,
    )


class TestSensitiveMatcher:
    def test_match_exact(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match("这是一段包含毒品的文本")
        assert len(results) == 1
        assert results[0].word == "毒品"
        assert results[0].category == "illegal_drugs"

    def test_match_multiple(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match("毒品和赌博都是违法的")
        categories = {r.word for r in results}
        assert "毒品" in categories
        assert "赌博" in categories

    def test_no_match(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match("这是一段正常的文本")
        assert len(results) == 0

    def test_overlapping_match(self, config):
        """重叠敏感词应全部匹配"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("中国,location\n中国人,location")
            path = f.name

        cfg = TextAnalysisConfig(sensitive_word_list_path=path, sensitive_fuzzy_match_enabled=False)
        matcher = SensitiveMatcher(cfg)
        results = matcher.match("中国人")
        assert len(results) == 2
        Path(path).unlink(missing_ok=True)

    def test_match_all_without_fuzzy(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match_all("毒品赌博")
        assert len(results) == 2

    def test_empty_word_list(self):
        """空词表不抛异常"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("")
            path = f.name

        cfg = TextAnalysisConfig(sensitive_word_list_path=path, sensitive_fuzzy_match_enabled=False)
        matcher = SensitiveMatcher(cfg)
        results = matcher.match("任何文本")
        assert len(results) == 0
        Path(path).unlink(missing_ok=True)
