# src/illegal_review/preprocessing_layer/context.py
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import numpy as np

from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.data_models import VideoMetadata, OCRResult


@dataclass
class SampledFrame:
    """采样帧（内部使用 np.ndarray，不对外暴露）"""
    frame_index: int
    timestamp: float       # 秒
    data: np.ndarray       # 图像数据 (H, W, C)


@dataclass
class PipelineContext:
    """Pipeline 阶段间共享上下文"""
    input_id: UUID
    video_path: str
    config: PreprocessingConfig

    # 从 InputResult 携带（透传，不重复 ffprobe）
    _input_metadata: Optional[VideoMetadata] = None

    # 解码输出
    raw_frames: Optional["FrameStore"] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None

    # 音频输出
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None

    # 采样输出
    sampled_frames: Optional[list[SampledFrame]] = None

    # 识别输出
    transcript: Optional[str] = None
    transcript_segments: Optional[list[dict]] = None
    ocr_results: Optional[list[OCRResult]] = None

    # 统计与错误
    stats: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
