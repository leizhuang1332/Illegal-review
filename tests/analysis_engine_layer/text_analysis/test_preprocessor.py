import pytest
from src.illegal_review.analysis_engine_layer.text_analysis.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_clean_html(self, preprocessor):
        result = preprocessor.clean("<p>hello</p>world")
        assert result == "hello world"

    def test_clean_special_chars(self, preprocessor):
        result = preprocessor.clean("hello!@#$world")
        assert result == "hello world"

    def test_clean_extra_spaces(self, preprocessor):
        result = preprocessor.clean("hello   world")
        assert result == "hello world"

    def test_segment(self, preprocessor):
        tokens = preprocessor.segment("我爱北京天安门")
        assert len(tokens) > 0
        assert "天安门" in tokens

    def test_normalize_traditional_to_simple(self, preprocessor):
        result = preprocessor.normalize("簡體轉繁體")
        assert "简体" in result

    def test_normalize_fullwidth_to_halfwidth(self, preprocessor):
        result = preprocessor.normalize("ＨＥＬＬＯ")
        assert result == "HELLO"

    def test_process_returns_preprocessed_text(self, preprocessor):
        result = preprocessor.process("  Hello World!  ")
        assert result.text == "Hello World"
        assert len(result.tokens) > 0
