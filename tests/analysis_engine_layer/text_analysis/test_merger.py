import pytest
from uuid import uuid4
from src.illegal_review.analysis_engine_layer.text_analysis.merger import ResultMerger
from src.illegal_review.data_models import (
    SourceAnalysis, SensitiveWord, CategoryResult, TextCategory,
)


class TestResultMerger:
    @pytest.fixture
    def merger(self):
        return ResultMerger()

    @pytest.fixture
    def video_id(self):
        return uuid4()

    def test_merge_both_sources(self, merger, video_id):
        ocr = SourceAnalysis(
            source="ocr", text_length=10,
            sensitive_words=[
                SensitiveWord(word="广告", start_pos=0, end_pos=2, match_type="exact", category="ad"),
            ],
            category=CategoryResult(category=TextCategory.AD, confidence=0.9, scores={"ad": 0.9}),
        )
        transcript = SourceAnalysis(
            source="transcript", text_length=20,
            sensitive_words=[],
            category=CategoryResult(category=TextCategory.NORMAL, confidence=0.95, scores={"normal": 0.95}),
        )

        result = merger.merge(ocr, transcript, video_id)
        assert result.ocr is not None
        assert result.transcript is not None
        assert len(result.violations) >= 1

    def test_merge_ocr_only(self, merger, video_id):
        ocr = SourceAnalysis(source="ocr", text_length=5)
        result = merger.merge(ocr, None, video_id)
        assert result.ocr is not None
        assert result.transcript is None

    def test_merge_transcript_only(self, merger, video_id):
        transcript = SourceAnalysis(source="transcript", text_length=5)
        result = merger.merge(None, transcript, video_id)
        assert result.ocr is None
        assert result.transcript is not None

    def test_merge_both_none(self, merger, video_id):
        result = merger.merge(None, None, video_id)
        assert result.ocr is None
        assert result.transcript is None
        assert result.violations == []

    def test_deduplicate_same_category(self, merger, video_id):
        ocr = SourceAnalysis(
            source="ocr", text_length=5,
            sensitive_words=[
                SensitiveWord(word="毒品", start_pos=0, end_pos=2, match_type="exact", category="drug"),
                SensitiveWord(word="毒品", start_pos=0, end_pos=2, match_type="exact", category="drug"),
            ],
        )
        result = merger.merge(ocr, None, video_id)
        drug_violations = [v for v in result.violations if v.category == "drug"]
        assert len(drug_violations) <= 1
