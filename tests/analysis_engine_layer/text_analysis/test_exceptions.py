"""
分析引擎层 - 文本分析异常体系单元测试

测试覆盖：
    1. TextAnalysisError — 基础异常类
    2. TextPreprocessingError — 文本预处理失败异常
    3. SemanticEncodingError — 语义编码失败异常
    4. ClassificationError — 文本分类失败异常
    5. 所有异常继承自 TextAnalysisError
"""

import pytest
from src.illegal_review.analysis_engine_layer.text_analysis.exceptions import (
    TextAnalysisError,
    TextPreprocessingError,
    SemanticEncodingError,
    ClassificationError,
)


class TestTextAnalysisExceptions:
    """文本分析异常体系测试"""

    def test_text_analysis_error_base(self):
        """TextAnalysisError 应能被引发并匹配错误消息"""
        with pytest.raises(TextAnalysisError, match="base error"):
            raise TextAnalysisError("base error")

    def test_preprocessing_error_subclass(self):
        """TextPreprocessingError 是 TextAnalysisError 的子类"""
        with pytest.raises(TextAnalysisError):
            raise TextPreprocessingError("preprocess failed")

    def test_semantic_encoding_error_subclass(self):
        """SemanticEncodingError 是 TextAnalysisError 的子类"""
        with pytest.raises(TextAnalysisError):
            raise SemanticEncodingError("encoding failed")

    def test_classification_error_subclass(self):
        """ClassificationError 是 TextAnalysisError 的子类"""
        with pytest.raises(TextAnalysisError):
            raise ClassificationError("classify failed")

    def test_all_inherit_from_text_analysis_error(self):
        """验证所有异常类都继承自 TextAnalysisError"""
        assert issubclass(TextPreprocessingError, TextAnalysisError)
        assert issubclass(SemanticEncodingError, TextAnalysisError)
        assert issubclass(ClassificationError, TextAnalysisError)
