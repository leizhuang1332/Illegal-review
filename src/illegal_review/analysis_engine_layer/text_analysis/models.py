from dataclasses import dataclass
from typing import List


@dataclass
class PreprocessedText:
    """预处理后的文本（清洗+分词结果）"""
    text: str           # 清洗后的完整文本
    tokens: List[str]   # 分词结果
