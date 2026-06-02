"""
分析引擎层 - 文本分析数据模型单元测试

测试覆盖：
    1. TextCategory — 枚举成员、值比较、JSON 序列化
    2. SensitiveWord — 默认值、字段赋值、model_dump()
    3. Entity — 默认值、字段赋值、边界情况（空名称、负置信度）
    4. CategoryResult — 全部 6 种类别、置信度范围、scores 字典
    5. SourceAnalysis — 全部字段、默认 None 行为、errors 列表累积
    6. TextAnalysisResult — 新结构（ocr/transcript/violations）
"""

import json
import pytest
from uuid import UUID, uuid4
from enum import Enum
from typing import Optional, List

from src.illegal_review.data_models import (
    # 文本分类枚举
    TextCategory,
    # 文本分析模型
    SensitiveWord,
    Entity,
    CategoryResult,
    SourceAnalysis,
    TextAnalysisResult,
    ViolationDetection,
)


# ============================================================
# TextCategory
# ============================================================

class TestTextCategory:
    """TextCategory 枚举成员、值比较、JSON 序列化"""

    def test_enum_members_count(self):
        """应包含 6 个类别成员"""
        members = list(TextCategory)
        assert len(members) == 6

    def test_enum_members_values(self):
        """验证所有成员的 name 和 value"""
        assert TextCategory.PORN.value == "porn"
        assert TextCategory.VIOLENCE.value == "violence"
        assert TextCategory.POLITICAL.value == "political"
        assert TextCategory.AD.value == "ad"
        assert TextCategory.COPYRIGHT.value == "copyright"
        assert TextCategory.NORMAL.value == "normal"

    def test_enum_is_str_subclass(self):
        """TextCategory 是 str+Enum，成员本身是字符串"""
        assert issubclass(TextCategory, str)
        assert issubclass(TextCategory, Enum)

    def test_enum_value_comparison_with_string(self):
        """枚举值可以直接与字符串比较"""
        assert TextCategory.PORN == "porn"
        assert TextCategory.VIOLENCE == "violence"
        assert TextCategory.NORMAL == "normal"

    def test_enum_json_serialization(self):
        """json.dumps 可直接序列化（str 子类）"""
        data = {"category": TextCategory.POLITICAL}
        dumped = json.dumps(data)
        assert dumped == '{"category": "political"}'

    def test_enum_from_value(self):
        """可以通过值反向获取枚举成员"""
        assert TextCategory("porn") is TextCategory.PORN
        assert TextCategory("normal") is TextCategory.NORMAL
        assert TextCategory("copyright") is TextCategory.COPYRIGHT

    def test_enum_invalid_value_raises(self):
        """无效值应抛出 ValueError"""
        with pytest.raises(ValueError):
            TextCategory("invalid_category")

    def test_enum_all_values_unique(self):
        """所有枚举值唯一"""
        values = [m.value for m in TextCategory]
        assert len(values) == len(set(values))


# ============================================================
# SensitiveWord
# ============================================================

class TestSensitiveWord:
    """SensitiveWord 默认值、字段赋值、model_dump()"""

    def test_required_fields_only(self):
        """仅提供必填字段应正常创建"""
        sw = SensitiveWord(
            word="赌博",
            start_pos=0,
            end_pos=2,
            match_type="exact",
            category="gambling",
        )
        assert sw.word == "赌博"
        assert sw.start_pos == 0
        assert sw.end_pos == 2
        assert sw.match_type == "exact"
        assert sw.category == "gambling"

    def test_field_types(self):
        """验证各字段类型正确"""
        sw = SensitiveWord(
            word="test", start_pos=5, end_pos=9,
            match_type="fuzzy", category="violence",
        )
        assert isinstance(sw.word, str)
        assert isinstance(sw.start_pos, int)
        assert isinstance(sw.end_pos, int)
        assert isinstance(sw.match_type, str)
        assert isinstance(sw.category, str)

    def test_model_dump_roundtrip(self):
        """model_dump() 输出可重新加载"""
        sw = SensitiveWord(
            word="敏感词", start_pos=10, end_pos=13,
            match_type="regex", category="porn",
        )
        dumped = sw.model_dump()
        assert dumped["word"] == "敏感词"
        assert dumped["start_pos"] == 10
        assert dumped["end_pos"] == 13
        assert dumped["match_type"] == "regex"
        assert dumped["category"] == "porn"

        # 反序列化验证
        sw2 = SensitiveWord.model_validate(dumped)
        assert sw2.word == sw.word
        assert sw2.match_type == sw.match_type

    def test_zero_length_word(self):
        """空字符串敏感词"""
        sw = SensitiveWord(
            word="", start_pos=0, end_pos=0,
            match_type="exact", category="other",
        )
        assert sw.word == ""

    def test_all_match_types(self):
        """验证三种 match_type 均可接受"""
        for mt in ("exact", "fuzzy", "regex"):
            sw = SensitiveWord(
                word="x", start_pos=0, end_pos=1,
                match_type=mt, category="ad",
            )
            assert sw.match_type == mt


# ============================================================
# Entity
# ============================================================

class TestEntity:
    """Entity 默认值、字段赋值、边界情况"""

    def test_required_fields(self):
        """仅提供必填字段应正常创建"""
        ent = Entity(
            name="张三", type="person",
            start_pos=0, end_pos=2, confidence=0.95,
        )
        assert ent.name == "张三"
        assert ent.type == "person"
        assert ent.start_pos == 0
        assert ent.end_pos == 2
        assert ent.confidence == 0.95

    def test_confidence_zero(self):
        """置信度为 0 应可接受"""
        ent = Entity(
            name="未知", type="other",
            start_pos=0, end_pos=2, confidence=0.0,
        )
        assert ent.confidence == 0.0

    def test_confidence_one(self):
        """置信度为 1 应可接受"""
        ent = Entity(
            name="某人", type="person",
            start_pos=0, end_pos=2, confidence=1.0,
        )
        assert ent.confidence == 1.0

    def test_all_entity_types(self):
        """验证所有五种 entity type 均可接受"""
        for t in ("person", "location", "organization", "time", "other"):
            ent = Entity(
                name="test", type=t,
                start_pos=0, end_pos=4, confidence=0.5,
            )
            assert ent.type == t

    def test_empty_name(self):
        """实体名称为空字符串"""
        ent = Entity(
            name="", type="other",
            start_pos=0, end_pos=0, confidence=0.0,
        )
        assert ent.name == ""

    def test_negative_confidence(self):
        """负置信度当前不应被验证拦截（如无 ge=0 约束）"""
        ent = Entity(
            name="test", type="other",
            start_pos=0, end_pos=4, confidence=-0.5,
        )
        # 注意：当前模型未加 ge=0 约束，所以 -0.5 可通过
        # 若未来添加约束，此处应相应调整
        assert ent.confidence == -0.5

    def test_large_positions(self):
        """较大的起始/结束位置"""
        ent = Entity(
            name="long text entity", type="other",
            start_pos=10000, end_pos=10018, confidence=0.8,
        )
        assert ent.start_pos == 10000
        assert ent.end_pos == 10018

    def test_model_dump_roundtrip(self):
        """model_dump() 可逆"""
        ent = Entity(
            name="北京", type="location",
            start_pos=5, end_pos=7, confidence=0.92,
        )
        dumped = ent.model_dump()
        assert dumped["name"] == "北京"
        assert dumped["type"] == "location"

        ent2 = Entity.model_validate(dumped)
        assert ent2.name == ent.name
        assert ent2.confidence == ent.confidence


# ============================================================
# CategoryResult
# ============================================================

class TestCategoryResult:
    """CategoryResult 全部 6 种类别、置信度范围、scores 字典"""

    def test_required_fields_only(self):
        """仅提供必填字段"""
        cr = CategoryResult(
            category=TextCategory.PORN,
            confidence=0.95,
            scores={"porn": 0.95, "normal": 0.05},
        )
        assert cr.category is TextCategory.PORN
        assert cr.confidence == 0.95
        assert cr.scores["porn"] == 0.95
        assert cr.scores["normal"] == 0.05

    def test_all_six_categories(self):
        """验证全部 6 种 TextCategory 均可作为分类结果"""
        for cat in TextCategory:
            cr = CategoryResult(
                category=cat,
                confidence=1.0,
                scores={cat.value: 1.0},
            )
            assert cr.category == cat
            assert cr.category.value == cat.value

    def test_confidence_range_zero_to_one(self):
        """置信度应在 0~1 范围内"""
        cr_low = CategoryResult(
            category=TextCategory.NORMAL,
            confidence=0.0,
            scores={"normal": 0.0},
        )
        cr_high = CategoryResult(
            category=TextCategory.NORMAL,
            confidence=1.0,
            scores={"normal": 1.0},
        )
        assert cr_low.confidence == 0.0
        assert cr_high.confidence == 1.0

    def test_scores_dict_multiple_categories(self):
        """scores 包含所有类别的概率分布"""
        scores = {
            "porn": 0.01,
            "violence": 0.02,
            "political": 0.82,
            "ad": 0.05,
            "copyright": 0.05,
            "normal": 0.05,
        }
        cr = CategoryResult(
            category=TextCategory.POLITICAL,
            confidence=0.82,
            scores=scores,
        )
        assert len(cr.scores) == 6
        assert abs(sum(cr.scores.values()) - 1.0) < 1e-6
        assert cr.scores["political"] == 0.82

    def test_confidence_above_one(self):
        """置信度 >1 当前不应被拦截（如无 le=1 约束）"""
        cr = CategoryResult(
            category=TextCategory.NORMAL,
            confidence=1.5,
            scores={"normal": 1.5},
        )
        assert cr.confidence == 1.5

    def test_string_category_acceptance(self):
        """category 字段接受字符串值（TextCategory 是 str+Enum）"""
        cr = CategoryResult(
            category="porn",
            confidence=0.9,
            scores={"porn": 0.9, "normal": 0.1},
        )
        assert cr.category == TextCategory.PORN
        assert cr.category == "porn"

    def test_model_dump_roundtrip(self):
        """model_dump() 输出可反序列化"""
        cr = CategoryResult(
            category=TextCategory.VIOLENCE,
            confidence=0.88,
            scores={"violence": 0.88, "normal": 0.12},
        )
        dumped = cr.model_dump()
        assert dumped["category"] == "violence"
        assert dumped["confidence"] == 0.88

        cr2 = CategoryResult.model_validate(dumped)
        assert cr2.category == cr.category
        assert cr2.confidence == cr.confidence


# ============================================================
# SourceAnalysis
# ============================================================

class TestSourceAnalysis:
    """SourceAnalysis 全部字段、默认 None 行为、errors 列表累积"""

    def test_required_fields_only(self):
        """仅提供必填字段，默认值正常"""
        sa = SourceAnalysis(source="ocr", text_length=100)
        assert sa.source == "ocr"
        assert sa.text_length == 100
        assert sa.semantic_embedding is None
        assert sa.sensitive_words == []
        assert sa.sentiment_score is None
        assert sa.entities == []
        assert sa.category is None
        assert sa.errors == []

    def test_both_source_values(self):
        """验证 source 字段 ocr 和 transcript"""
        sa_ocr = SourceAnalysis(source="ocr", text_length=50)
        sa_tr = SourceAnalysis(source="transcript", text_length=200)
        assert sa_ocr.source == "ocr"
        assert sa_tr.source == "transcript"

    def test_text_length_zero(self):
        """文本长度为 0 的边界情况"""
        sa = SourceAnalysis(source="ocr", text_length=0)
        assert sa.text_length == 0

    def test_semantic_embedding_assignment(self):
        """semantic_embedding 赋值（768 维向量）"""
        embedding = [float(i) for i in range(768)]
        sa = SourceAnalysis(
            source="transcript",
            text_length=100,
            semantic_embedding=embedding,
        )
        assert len(sa.semantic_embedding) == 768
        assert sa.semantic_embedding[0] == 0.0
        assert sa.semantic_embedding[767] == 767.0

    def test_semantic_embedding_default_none(self):
        """semantic_embedding 默认 None"""
        sa = SourceAnalysis(source="ocr", text_length=10)
        assert sa.semantic_embedding is None

    def test_sensitive_words_accumulation(self):
        """sensitive_words 列表累积"""
        sa = SourceAnalysis(source="ocr", text_length=50)
        assert len(sa.sensitive_words) == 0

        sa.sensitive_words.append(
            SensitiveWord(
                word="赌", start_pos=0, end_pos=1,
                match_type="exact", category="gambling",
            )
        )
        assert len(sa.sensitive_words) == 1

        sa.sensitive_words.append(
            SensitiveWord(
                word="毒", start_pos=2, end_pos=3,
                match_type="exact", category="drug",
            )
        )
        assert len(sa.sensitive_words) == 2
        assert sa.sensitive_words[0].word == "赌"
        assert sa.sensitive_words[1].word == "毒"

    def test_sentiment_score_range(self):
        """情感分数可接受 -1~1 范围内的值"""
        sa_neg = SourceAnalysis(source="ocr", text_length=10, sentiment_score=-1.0)
        sa_pos = SourceAnalysis(source="ocr", text_length=10, sentiment_score=1.0)
        sa_zero = SourceAnalysis(source="ocr", text_length=10, sentiment_score=0.0)
        assert sa_neg.sentiment_score == -1.0
        assert sa_pos.sentiment_score == 1.0
        assert sa_zero.sentiment_score == 0.0

    def test_sentiment_score_out_of_range(self):
        """情感分数超出 -1~1 当前不应被拦截（如无 ge=-1, le=1 约束）"""
        sa = SourceAnalysis(source="ocr", text_length=10, sentiment_score=2.5)
        assert sa.sentiment_score == 2.5

    def test_sentiment_score_default_none(self):
        """sentiment_score 默认 None"""
        sa = SourceAnalysis(source="ocr", text_length=10)
        assert sa.sentiment_score is None

    def test_entities_accumulation(self):
        """entities 列表累积"""
        sa = SourceAnalysis(source="transcript", text_length=200)
        assert len(sa.entities) == 0

        sa.entities.append(
            Entity(name="张三", type="person", start_pos=0, end_pos=2, confidence=0.95)
        )
        sa.entities.append(
            Entity(name="北京", type="location", start_pos=10, end_pos=12, confidence=0.90)
        )
        assert len(sa.entities) == 2
        assert sa.entities[0].name == "张三"
        assert sa.entities[1].name == "北京"

    def test_category_assignment(self):
        """category 字段赋值"""
        cr = CategoryResult(
            category=TextCategory.POLITICAL,
            confidence=0.85,
            scores={"political": 0.85, "normal": 0.15},
        )
        sa = SourceAnalysis(
            source="ocr", text_length=100, category=cr,
        )
        assert sa.category is not None
        assert sa.category.category == TextCategory.POLITICAL
        assert sa.category.confidence == 0.85

    def test_category_default_none(self):
        """category 默认 None（未分类时）"""
        sa = SourceAnalysis(source="ocr", text_length=10)
        assert sa.category is None

    def test_errors_list_accumulate(self):
        """errors 列表可累积多条错误"""
        sa = SourceAnalysis(source="ocr", text_length=50)
        sa.errors.append("OCR API timeout")
        sa.errors.append("Insufficient text content")
        assert len(sa.errors) == 2
        assert sa.errors[0] == "OCR API timeout"
        assert sa.errors[1] == "Insufficient text content"

    def test_errors_default_empty(self):
        """errors 默认为空列表"""
        sa = SourceAnalysis(source="ocr", text_length=50)
        assert sa.errors == []

    def test_sensitive_words_default_empty(self):
        """sensitive_words 默认为空列表"""
        sa = SourceAnalysis(source="transcript", text_length=100)
        assert sa.sensitive_words == []

    def test_entities_default_empty(self):
        """entities 默认为空列表"""
        sa = SourceAnalysis(source="transcript", text_length=100)
        assert sa.entities == []

    def test_model_dump_roundtrip(self):
        """model_dump() 可逆"""
        sa = SourceAnalysis(
            source="ocr",
            text_length=42,
            sentiment_score=0.3,
            sensitive_words=[
                SensitiveWord(
                    word="广告", start_pos=5, end_pos=7,
                    match_type="fuzzy", category="ad",
                )
            ],
            entities=[
                Entity(name="某公司", type="organization", start_pos=0, end_pos=3, confidence=0.8),
            ],
            category=CategoryResult(
                category=TextCategory.AD,
                confidence=0.75,
                scores={"ad": 0.75, "normal": 0.25},
            ),
            errors=["warning: short text"],
        )
        dumped = sa.model_dump()
        assert dumped["source"] == "ocr"
        assert dumped["text_length"] == 42
        assert len(dumped["sensitive_words"]) == 1
        assert dumped["category"]["category"] == "ad"
        assert len(dumped["errors"]) == 1

        sa2 = SourceAnalysis.model_validate(dumped)
        assert sa2.source == sa.source
        assert sa2.text_length == sa.text_length
        assert sa2.sensitive_words[0].word == "广告"
        assert sa2.category.category == TextCategory.AD
        assert sa2.errors[0] == "warning: short text"

    def test_full_construct_all_fields(self):
        """构造一个包含所有字段的完整实例"""
        sa = SourceAnalysis(
            source="transcript",
            text_length=500,
            semantic_embedding=[0.1] * 768,
            sensitive_words=[
                SensitiveWord(word="暴力", start_pos=10, end_pos=12, match_type="exact", category="violence"),
            ],
            sentiment_score=-0.5,
            entities=[
                Entity(name="某人", type="person", start_pos=0, end_pos=2, confidence=0.9),
            ],
            category=CategoryResult(
                category=TextCategory.VIOLENCE,
                confidence=0.9,
                scores={"violence": 0.9, "normal": 0.1},
            ),
            errors=[],
        )
        assert sa.source == "transcript"
        assert sa.text_length == 500
        assert len(sa.semantic_embedding) == 768
        assert len(sa.sensitive_words) == 1
        assert sa.sentiment_score == -0.5
        assert len(sa.entities) == 1
        assert sa.category is not None
        assert sa.errors == []


# ============================================================
# TextAnalysisResult
# ============================================================

class TestTextAnalysisResult:
    """TextAnalysisResult 新结构（ocr/transcript/violations）"""

    def test_required_fields_only(self):
        """仅提供必填字段 video_id"""
        result = TextAnalysisResult(video_id=uuid4())
        assert result.ocr is None
        assert result.transcript is None
        assert result.violations == []
        assert result.processing_stats == {}

    def test_video_id_type(self):
        """video_id 为 UUID 类型"""
        vid = uuid4()
        result = TextAnalysisResult(video_id=vid)
        assert isinstance(result.video_id, UUID)
        assert result.video_id == vid

    def test_ocr_and_transcript_separate(self):
        """ocr 和 transcript 可独立设置"""
        vid = uuid4()
        ocr_sa = SourceAnalysis(source="ocr", text_length=100)
        transcript_sa = SourceAnalysis(source="transcript", text_length=200)

        result_ocr = TextAnalysisResult(video_id=vid, ocr=ocr_sa)
        assert result_ocr.ocr is not None
        assert result_ocr.ocr.source == "ocr"
        assert result_ocr.transcript is None

        result_both = TextAnalysisResult(
            video_id=vid,
            ocr=ocr_sa,
            transcript=transcript_sa,
        )
        assert result_both.ocr.source == "ocr"
        assert result_both.transcript.source == "transcript"

    def test_violations_accumulation(self):
        """violations 列表可累积"""
        vid = uuid4()
        result = TextAnalysisResult(video_id=vid)
        assert len(result.violations) == 0

        result.violations.append(
            ViolationDetection(
                type="text",
                category="porn",
                confidence=0.95,
                timestamp=10.5,
                context="包含色情词汇",
                evidence={"matched_words": ["色情词1"]},
            )
        )
        assert len(result.violations) == 1

        result.violations.append(
            ViolationDetection(
                type="text",
                category="violence",
                confidence=0.80,
                timestamp=20.0,
                context="暴力内容",
                evidence={"detail": "检测到暴力词汇"},
            )
        )
        assert len(result.violations) == 2
        assert result.violations[0].category == "porn"
        assert result.violations[1].category == "violence"

    def test_violations_default_empty(self):
        """violations 默认空列表"""
        result = TextAnalysisResult(video_id=uuid4())
        assert result.violations == []

    def test_processing_stats(self):
        """processing_stats 字典赋值"""
        result = TextAnalysisResult(video_id=uuid4())
        assert result.processing_stats == {}

        result.processing_stats["ocr_duration_ms"] = 150.5
        result.processing_stats["transcript_duration_ms"] = 320.0
        result.processing_stats["total_duration_ms"] = 470.5

        assert result.processing_stats["ocr_duration_ms"] == 150.5
        assert result.processing_stats["total_duration_ms"] == 470.5

    def test_full_construct(self):
        """构造一个包含所有字段的完整实例"""
        vid = uuid4()
        ocr_sa = SourceAnalysis(
            source="ocr",
            text_length=300,
            sensitive_words=[
                SensitiveWord(word="赌", start_pos=0, end_pos=1, match_type="exact", category="gambling"),
            ],
            category=CategoryResult(
                category=TextCategory.PORN,
                confidence=0.88,
                scores={"porn": 0.88, "normal": 0.12},
            ),
        )
        transcript_sa = SourceAnalysis(
            source="transcript",
            text_length=500,
            sentiment_score=0.2,
        )
        violations_list = [
            ViolationDetection(
                type="text", category="porn", confidence=0.88,
                timestamp=5.0, context="OCR 检测到敏感词",
                evidence={"word": "赌", "position": 0},
            ),
        ]
        stats = {"ocr_duration_ms": 100.0, "nlp_duration_ms": 50.0}

        result = TextAnalysisResult(
            video_id=vid,
            ocr=ocr_sa,
            transcript=transcript_sa,
            violations=violations_list,
            processing_stats=stats,
        )

        assert result.video_id == vid
        assert result.ocr.source == "ocr"
        assert result.transcript.source == "transcript"
        assert len(result.violations) == 1
        assert result.violations[0].category == "porn"
        assert result.processing_stats["ocr_duration_ms"] == 100.0

    def test_model_dump_roundtrip(self):
        """model_dump() 可逆"""
        vid = uuid4()
        result = TextAnalysisResult(
            video_id=vid,
            ocr=SourceAnalysis(source="ocr", text_length=50),
            violations=[
                ViolationDetection(
                    type="text", category="ad", confidence=0.7,
                    timestamp=5.0, context="广告内容", evidence={"keywords": ["广告"]},
                ),
            ],
            processing_stats={"total_ms": 200.0},
        )
        dumped = result.model_dump()
        assert dumped["video_id"] == vid
        assert dumped["ocr"]["source"] == "ocr"
        assert len(dumped["violations"]) == 1
        assert dumped["processing_stats"]["total_ms"] == 200.0

        result2 = TextAnalysisResult.model_validate(dumped)
        assert result2.video_id == vid
        assert result2.ocr.text_length == 50
        assert result2.violations[0].confidence == 0.7

    def test_result_no_analysis(self):
        """没有任何分析结果时的默认状态"""
        result = TextAnalysisResult(video_id=uuid4())
        assert result.ocr is None
        assert result.transcript is None
        assert result.violations == []
        assert result.processing_stats == {}
