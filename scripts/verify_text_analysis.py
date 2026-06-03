"""
文本分析引擎验证脚本

用法:
    python scripts/verify_text_analysis.py

如未下载 BERT 模型，首次运行需联网（约 400MB），后续缓存到本地。
首次可能耗时 1-2 分钟（模型下载 + 加载）。
"""

import asyncio
import logging
import sys
import os
from uuid import uuid4

# 将项目根目录添加到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 禁用 transformers 的 INFO 日志，保持输出清爽
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.analysis_engine_layer.text_analysis import TextAnalysisService
from src.illegal_review.data_models import OCRResult, TextCategory


async def main():
    print("=" * 60)
    print("文本分析引擎验证")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 构造 Service（构造时预加载 BERT + NER + 分类模型，首次约 1-2 分钟）
    # ------------------------------------------------------------------
    print("\n[1/4] 加载文本分析引擎（首次会下载模型）...")
    sys.stdout.flush()

    config = TextAnalysisConfig(
        sensitive_word_list_path="config/sensitive_words.txt",
        sensitive_fuzzy_match_enabled=False,
        ocr_confidence_threshold=0.5,
        ner_enabled=True,
    )
    service = TextAnalysisService(config)
    print("     引擎加载完成 ✓")

    # ------------------------------------------------------------------
    # 2. 场景 A：OCR 含敏感词 + 语音正常
    # ------------------------------------------------------------------
    print("\n[2/4] 场景 A：OCR 含敏感词 + 语音正常")
    sys.stdout.flush()

    result_a = await service.analyze_all(
        video_id=uuid4(),
        ocr_results=[
            OCRResult(text="免费领取毒品", confidence=0.95, bbox=None, frame_index=0),
            OCRResult(text="联系客服微信xxx", confidence=0.80, bbox=None, frame_index=1),
        ],
        transcript="今天我们来讨论一下正常的内容审核流程和技术方案",
    )

    print(f"  OCR 敏感词: {[w.word for w in result_a.ocr.sensitive_words]}")
    print(f"  OCR 分类: {result_a.ocr.category.category.value if result_a.ocr.category else 'N/A'}")
    print(f"  OCR 情感分: {result_a.ocr.sentiment_score}")
    print(f"  OCR 实体: {[e.name for e in result_a.ocr.entities]}")
    print(f"  语音敏感词: {[w.word for w in result_a.transcript.sensitive_words]}")
    print(f"  语音分类: {result_a.transcript.category.category.value if result_a.transcript.category else 'N/A'}")
    print(f"  违规汇总: {len(result_a.violations)} 条")

    # ------------------------------------------------------------------
    # 3. 场景 B：仅 OCR（无语音）
    # ------------------------------------------------------------------
    print("\n[3/4] 场景 B：仅 OCR 文本（无语音）")
    sys.stdout.flush()

    result_b = await service.analyze_all(
        video_id=uuid4(),
        ocr_results=[
            OCRResult(text="赌博网站上线啦", confidence=0.90, bbox=None, frame_index=0),
        ],
        transcript=None,
    )

    print(f"  OCR 敏感词: {[w.word for w in result_b.ocr.sensitive_words]}")
    print(f"  语音结果: {result_b.transcript}")  # 应为 None
    print(f"  违规汇总: {len(result_b.violations)} 条")

    # ------------------------------------------------------------------
    # 4. 场景 C：仅语音（无 OCR）
    # ------------------------------------------------------------------
    print("\n[4/4] 场景 C：仅语音文本（无 OCR）")
    sys.stdout.flush()

    result_c = await service.analyze_all(
        video_id=uuid4(),
        ocr_results=None,
        transcript="我非常愤怒，这个平台太糟糕了，全是骗子",
    )

    print(f"  OCR 结果: {result_c.ocr}")  # 应为 None
    print(f"  语音敏感词: {[w.word for w in result_c.transcript.sensitive_words]}")
    print(f"  语音情感分: {result_c.transcript.sentiment_score}")
    print(f"  语音实体: {[e.name for e in result_c.transcript.entities]}")
    print(f"  语音分类: {result_c.transcript.category.category.value if result_c.transcript.category else 'N/A'}")
    print(f"  违规汇总: {len(result_c.violations)} 条")

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("验证完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
