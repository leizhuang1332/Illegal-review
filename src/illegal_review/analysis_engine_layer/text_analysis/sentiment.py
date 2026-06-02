from typing import Dict, Set

import jieba
from src.illegal_review.config.settings import TextAnalysisConfig


class SentimentAnalyzer:
    """情感分析 — 基于 lexicon + 规则（无模型依赖）"""

    def __init__(self, config: TextAnalysisConfig):
        self._positive_words: Set[str] = {
            "好", "喜欢", "爱", "优秀", "美丽", "漂亮", "棒", "赞",
            "开心", "高兴", "快乐", "幸福", "美好", "精彩", "完美",
            "出色", "享受", "舒适", "满意", "欣赏", "喜爱",
        }
        self._negative_words: Set[str] = {
            "坏", "差", "讨厌", "恨", "垃圾", "恶心", "丑陋",
            "伤心", "难过", "痛苦", "愤怒", "生气", "糟糕",
            "恐怖", "可怕", "严重", "危险", "失败", "崩溃",
            "不良", "违法", "违规", "非法",
        }
        self._negation_words: Set[str] = {
            "不", "没", "别", "勿", "毋", "未", "无", "不是",
            "没有", "不要", "不会", "不能", "不该",
        }
        self._adverb_boost: Dict[str, float] = {
            "非常": 1.5, "很": 1.3, "太": 1.4, "极其": 1.8,
            "特别": 1.5, "十分": 1.4, "相当": 1.3, "无比": 1.8,
            "有点": 0.7, "稍微": 0.6, "比较": 0.8, "不太": 0.5,
        }

    def analyze(self, text: str) -> float:
        """返回 -1.0 ~ 1.0，正值=正面，负值=负面，0=中性"""
        if not text or not text.strip():
            return 0.0

        words = jieba.lcut(text)
        score = 0.0
        negation_distance = 0
        current_weight = 1.0
        base = 0.5

        for w in words:
            if w in self._negation_words:
                negation_distance = 2
                continue

            if w in self._adverb_boost:
                current_weight = self._adverb_boost[w]
                continue

            if w in self._positive_words:
                if negation_distance > 0:
                    score -= current_weight * base * 0.5
                else:
                    score += current_weight * base
            elif w in self._negative_words:
                if negation_distance > 0:
                    score += current_weight * base * 0.5
                else:
                    score -= current_weight * base

            if negation_distance > 0:
                negation_distance -= 1
            current_weight = 1.0

        return max(-1.0, min(1.0, score))
