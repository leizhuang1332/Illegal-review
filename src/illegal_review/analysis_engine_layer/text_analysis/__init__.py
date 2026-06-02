"""
文本分析引擎模块

负责文本内容分析：
- BERT语义分析：理解文本语义
- 敏感词匹配：AC自动机检测敏感词汇
- 情感分析：基于词典分析文本情感倾向
- 实体识别：命名实体识别（人物/地点/组织/时间）
- 文本分类：6类违规文本分类（可扩展）

上游：预处理层（OCR文本 + 语音转写文本）
下游：规则引擎、AI引擎
"""

from src.illegal_review.analysis_engine_layer.text_analysis.service import TextAnalysisService

__all__ = ["TextAnalysisService"]
