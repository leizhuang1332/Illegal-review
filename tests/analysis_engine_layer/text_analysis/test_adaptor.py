import pytest
from uuid import uuid4
from src.illegal_review.analysis_engine_layer.text_analysis.adaptor import TextAdaptor
from src.illegal_review.data_models import OCRResult


class TestTextAdaptor:
    def test_extract_ocr_text_merges_multiple(self):
        results = [
            OCRResult(text="世界", confidence=0.9, bbox=None, frame_index=5),
            OCRResult(text="你好", confidence=0.95, bbox=None, frame_index=3),
        ]
        text = TextAdaptor.extract_ocr_text(results)
        assert text == "你好 世界"

    def test_extract_ocr_text_filters_low_confidence(self):
        results = [
            OCRResult(text="正常", confidence=0.9, bbox=None, frame_index=0),
            OCRResult(text="噪声", confidence=0.3, bbox=None, frame_index=1),
        ]
        text = TextAdaptor.extract_ocr_text(results, min_confidence=0.5)
        assert text == "正常"

    def test_extract_ocr_text_empty(self):
        assert TextAdaptor.extract_ocr_text([]) is None

    def test_extract_ocr_text_all_filtered(self):
        results = [
            OCRResult(text="噪声", confidence=0.2, bbox=None, frame_index=0),
        ]
        assert TextAdaptor.extract_ocr_text(results, min_confidence=0.5) is None

    def test_extract_ocr_text_max_length(self):
        results = [
            OCRResult(text="a" * 100, confidence=0.9, bbox=None, frame_index=0),
        ]
        text = TextAdaptor.extract_ocr_text(results, max_length=10)
        assert len(text) == 10

    def test_extract_transcript_normal(self):
        text = TextAdaptor.extract_transcript("Hello World")
        assert text == "Hello World"

    def test_extract_transcript_empty(self):
        assert TextAdaptor.extract_transcript("") is None
        assert TextAdaptor.extract_transcript(None) is None
        assert TextAdaptor.extract_transcript("  ") is None

    def test_extract_transcript_max_length(self):
        text = TextAdaptor.extract_transcript("a" * 200, max_length=50)
        assert len(text) == 50
