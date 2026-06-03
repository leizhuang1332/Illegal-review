"""
全流程集成测试：输入层 → 预处理层 → 文本分析引擎

用法 1：使用真实视频文件（全流程）
    python scripts/integration_test.py path/to/video.mp4

用法 2：使用模拟数据（仅文本分析引擎，不需要视频）
    python scripts/integration_test.py

用法 3：使用模拟数据但过预处理层
    python scripts/integration_test.py --mock-preprocess
"""

import argparse
import asyncio
import logging
import sys
import os
import time
from uuid import uuid4
from pathlib import Path

# 将项目根目录添加到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
# 屏蔽第三方库的 INFO 日志
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("whisper").setLevel(logging.WARNING)
logging.getLogger("easyocr").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)


async def run_with_video(video_path: str):
    """全流程：输入层 → 预处理层 → 文本分析引擎"""
    from src.illegal_review.input_layer.service import InputService
    from src.illegal_review.preprocessing_layer import PreprocessingService
    from src.illegal_review.analysis_engine_layer.text_analysis import TextAnalysisService
    from src.illegal_review.config.settings import SystemConfig

    cfg = SystemConfig()
    print(f"\n{'='*60}")
    print(f"[1/3] 输入层处理: {video_path}")
    print(f"{'='*60}")

    input_service = InputService(cfg.input_layer)
    file_path = Path(video_path)
    input_result = await input_service.handle_file_upload(file_path, file_path.name)
    print(f"  input_id: {input_result.input_id}")
    print(f"  状态: {input_result.status}")

    print(f"\n{'='*60}")
    print(f"[2/3] 预处理层: 解码 → 帧采样 → OCR → 语音转写")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    preprocess_service = PreprocessingService(cfg.preprocessing)
    preprocess_result = await preprocess_service.process(input_result)
    elapsed = time.perf_counter() - t0

    print(f"  处理耗时: {elapsed:.1f}s")
    print(f"  总帧数: {preprocess_result.metadata.duration * preprocess_result.metadata.fps:.0f}" if preprocess_result.metadata else "  元数据: None")
    print(f"  采样帧数: {len(preprocess_result.frames) if preprocess_result.frames else 0}")
    print(f"  有音频: {preprocess_result.audio is not None}")
    print(f"  语音转写: {'共 ' + str(len(preprocess_result.transcript or '')) + ' 字' if preprocess_result.transcript else '无'}")
    print(f"  OCR: {len(preprocess_result.ocr_results) if preprocess_result.ocr_results else 0} 条")

    print(f"\n{'='*60}")
    print(f"[3/3] 文本分析引擎: 敏感词 → 情感 → 语义 → NER → 分类")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    text_service = TextAnalysisService(cfg.text_analysis)
    text_result = await text_service.analyze_all(
        video_id=preprocess_result.input_id,
        ocr_results=preprocess_result.ocr_results,
        transcript=preprocess_result.transcript,
    )
    elapsed = time.perf_counter() - t0

    print(f"  分析耗时: {elapsed:.1f}s")

    _print_text_result(text_result)


async def run_mock():
    """仅文本分析引擎（使用模拟预处理数据）"""
    from src.illegal_review.config.settings import TextAnalysisConfig
    from src.illegal_review.analysis_engine_layer.text_analysis import TextAnalysisService
    from src.illegal_review.data_models import OCRResult, TextAnalysisResult

    print(f"\n{'='*60}")
    print(f"文本分析引擎 · 模拟数据验证")
    print(f"{'='*60}")

    config = TextAnalysisConfig(
        sensitive_word_list_path="config/sensitive_words.txt",
        sensitive_fuzzy_match_enabled=False,
    )

    print("\n[1/3] 加载文本分析引擎（含 BERT/NER/分类模型）...")
    t0 = time.perf_counter()
    service = TextAnalysisService(config)
    print(f"  加载耗时: {time.perf_counter() - t0:.1f}s ✓")

    print(f"\n[2/3] 执行分析...")
    t0 = time.perf_counter()
    result = await service.analyze_all(
        video_id=uuid4(),
        ocr_results=[
            OCRResult(text="免费领取毒品", confidence=0.95, bbox=None, frame_index=0),
            OCRResult(text="联系微信", confidence=0.80, bbox=None, frame_index=1),
            OCRResult(text="今晚一起打游戏", confidence=0.90, bbox=None, frame_index=2),
        ],
        transcript="我非常愤怒，这个平台全是骗人的，大家千万不要上当",
    )
    elapsed = time.perf_counter() - t0
    print(f"  分析耗时: {elapsed:.2f}s")

    print(f"\n[3/3] 分析结果")
    _print_text_result(result)


async def run_mock_preprocess():
    """过预处理层但使用模拟输入（验证层间数据传递）"""
    from src.illegal_review.config.settings import SystemConfig
    from src.illegal_review.preprocessing_layer import PreprocessingService
    from src.illegal_review.analysis_engine_layer.text_analysis import TextAnalysisService
    from src.illegal_review.data_models import (
        InputResult, SourceInfo, VideoMetadata,
    )
    from datetime import datetime, timezone

    cfg = SystemConfig()

    # 构造模拟 InputResult
    input_result = InputResult(
        input_id=uuid4(),
        input_type="file",
        source_info=SourceInfo(
            original_source="test.mp4",
            file_size=1024,
            content_type="video/mp4",
        ),
        video_metadata=VideoMetadata(
            duration=10.0, fps=30.0, width=1920, height=1080, codec="h264",
        ),
        temp_path="/tmp/test.mp4",
        status="completed",
        created_at=datetime.now(timezone.utc),
        processed_at=datetime.now(timezone.utc),
    )

    print(f"\n{'='*60}")
    print(f"通过预处理层模拟 · 数据流验证")
    print(f"{'='*60}")

    print("\n[1/2] 预处理层（模拟输入）...")
    preprocess_service = PreprocessingService(cfg.preprocessing)
    preprocess_result = await preprocess_service.process(input_result)

    print(f"  frames: {'有' if preprocess_result.frames else 'None'}")
    print(f"  audio: {'有' if preprocess_result.audio else 'None'}")
    print(f"  transcript: {'有' if preprocess_result.transcript else 'None'}")
    print(f"  ocr_results: {len(preprocess_result.ocr_results) if preprocess_result.ocr_results else 'None'}")

    print("\n[2/2] 文本分析引擎...")
    text_service = TextAnalysisService(cfg.text_analysis)
    text_result = await text_service.analyze_all(
        video_id=preprocess_result.input_id,
        ocr_results=preprocess_result.ocr_results,
        transcript=preprocess_result.transcript,
    )

    print(f"  OCR: {'有' if text_result.ocr else 'None'}")
    print(f"  语音: {'有' if text_result.transcript else 'None'}")
    print(f"  违规汇总: {len(text_result.violations)} 条")
    _print_text_result(text_result)


def _print_text_result(result: "TextAnalysisResult"):
    """打印文本分析结果"""
    print(f"\n  --- OCR 分析 ---")
    if result.ocr:
        print(f"  文本长度: {result.ocr.text_length}")
        print(f"  敏感词: {[f'{w.word}({w.category})' for w in result.ocr.sensitive_words]}")
        print(f"  情感分: {result.ocr.sentiment_score}")
        print(f"  实体: {[f'{e.name}({e.type})' for e in result.ocr.entities]}")
        print(f"  分类: {result.ocr.category.category.value if result.ocr.category else 'N/A'} "
              f"(置信度: {result.ocr.category.confidence:.2f})" if result.ocr.category else "")
        if result.ocr.errors:
            print(f"  降级记录: {result.ocr.errors}")
    else:
        print(f"  无 OCR 输入")

    print(f"\n  --- 语音分析 ---")
    if result.transcript:
        print(f"  文本长度: {result.transcript.text_length}")
        print(f"  敏感词: {[f'{w.word}({w.category})' for w in result.transcript.sensitive_words]}")
        print(f"  情感分: {result.transcript.sentiment_score}")
        print(f"  实体: {[f'{e.name}({e.type})' for e in result.transcript.entities]}")
        print(f"  分类: {result.transcript.category.category.value if result.transcript.category else 'N/A'} "
              f"(置信度: {result.transcript.category.confidence:.2f})" if result.transcript.category else "")
        if result.transcript.errors:
            print(f"  降级记录: {result.transcript.errors}")
    else:
        print(f"  无语音输入")

    print(f"\n  --- 违规检测汇总 ({len(result.violations)} 条) ---")
    for i, v in enumerate(result.violations):
        print(f"  {i+1}. [{v.type}] {v.category} (置信度: {v.confidence:.2f})")
        if v.evidence:
            src = v.evidence.get("source", "")
            word = v.evidence.get("word", "")
            print(f"     来源: {src}, 匹配: {word}")


def main():
    parser = argparse.ArgumentParser(description="文本分析引擎全流程集成测试")
    parser.add_argument("video", nargs="?", help="视频文件路径（全流程测试）")
    parser.add_argument("--mock-preprocess", action="store_true",
                        help="模拟预处理层数据（验证层间传递）")
    args = parser.parse_args()

    if args.video:
        asyncio.run(run_with_video(args.video))
    elif args.mock_preprocess:
        asyncio.run(run_mock_preprocess())
    else:
        asyncio.run(run_mock())


if __name__ == "__main__":
    main()
