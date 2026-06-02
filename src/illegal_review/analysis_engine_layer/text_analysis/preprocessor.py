import re
from typing import List, Optional, Set

import jieba

from src.illegal_review.analysis_engine_layer.text_analysis.models import PreprocessedText


class TextPreprocessor:
    """文本清洗与分词（使用 jieba，无额外模型依赖）"""

    _stopwords: Optional[Set[str]] = None

    def __init__(self, language: str = "zh"):
        self._jieba = jieba

    def clean(self, text: str) -> str:
        """去除HTML标签、特殊符号、多余空白"""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^\w\s一-鿿]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def segment(self, text: str) -> List[str]:
        """分词"""
        return self._jieba.lcut(text)

    def _get_stopwords(self) -> Set[str]:
        """加载停用词表（类级别缓存）"""
        if TextPreprocessor._stopwords is None:
            TextPreprocessor._stopwords = {
                "的", "了", "在", "是", "我", "有", "和", "就",
                "不", "人", "都", "一", "一个", "上", "也", "很",
                "到", "说", "要", "去", "你", "会", "着", "没有",
                "看", "好", "自己", "这", "他", "她", "它", "们",
            }
        return TextPreprocessor._stopwords

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """去停用词"""
        stopwords = self._get_stopwords()
        return [t for t in tokens if t not in stopwords]

    def normalize(self, text: str) -> str:
        """繁简转换、全角半角统一"""
        import opencc
        converter = opencc.OpenCC('t2s')
        text = converter.convert(text)
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    def process(self, text: str) -> PreprocessedText:
        """全流程：clean -> normalize -> segment -> remove_stopwords"""
        text = self.clean(text)
        text = self.normalize(text)
        tokens = self.segment(text)
        tokens = self.remove_stopwords(tokens)
        return PreprocessedText(text=text, tokens=tokens)
