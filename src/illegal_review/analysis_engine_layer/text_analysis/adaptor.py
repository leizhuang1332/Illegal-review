from typing import List, Optional

from src.illegal_review.data_models import OCRResult


class TextAdaptor:
    """将 PreprocessingResult 中的文本提取为标准字符串"""

    @staticmethod
    def extract_ocr_text(
        ocr_results: Optional[List[OCRResult]],
        min_confidence: float = 0.5,
        max_length: int = 5000,
    ) -> Optional[str]:
        """从 OCR 结果提取文本：排序 → 过滤置信度 → 合并 → 截断"""
        if not ocr_results:
            return None
        valid = [r for r in ocr_results if r.confidence >= min_confidence]
        if not valid:
            return None
        valid.sort(key=lambda r: r.frame_index)
        text = " ".join(r.text for r in valid)
        return text[:max_length]

    @staticmethod
    def extract_transcript(
        transcript: Optional[str],
        max_length: int = 10000,
    ) -> Optional[str]:
        """提取转写文本，仅做空值和长度处理"""
        if not transcript or not transcript.strip():
            return None
        return transcript[:max_length]
