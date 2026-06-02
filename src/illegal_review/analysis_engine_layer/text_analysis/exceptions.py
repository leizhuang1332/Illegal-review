"""
文本分析异常体系

定义文本分析引擎中使用的异常类层次结构：
    TextAnalysisError
    ├── TextPreprocessingError    — 文本预处理失败
    ├── SemanticEncodingError     — 语义编码失败
    └── ClassificationError       — 文本分类失败
"""


class TextAnalysisError(Exception):
    """文本分析基础异常"""
    pass


class TextPreprocessingError(TextAnalysisError):
    """文本预处理失败"""
    pass


class SemanticEncodingError(TextAnalysisError):
    """语义编码失败"""
    pass


class ClassificationError(TextAnalysisError):
    """文本分类失败"""
    pass
