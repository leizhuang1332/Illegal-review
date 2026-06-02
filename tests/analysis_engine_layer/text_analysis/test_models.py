from src.illegal_review.analysis_engine_layer.text_analysis.models import PreprocessedText


class TestPreprocessedText:
    def test_create(self):
        pt = PreprocessedText(text="hello world", tokens=["hello", "world"])
        assert pt.text == "hello world"
        assert pt.tokens == ["hello", "world"]

    def test_empty(self):
        pt = PreprocessedText(text="", tokens=[])
        assert pt.text == ""
        assert pt.tokens == []
