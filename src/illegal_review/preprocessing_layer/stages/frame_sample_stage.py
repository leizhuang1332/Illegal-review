import logging
import numpy as np
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext, SampledFrame
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class FrameSampleStage(Stage):
    """帧采样 + 场景变化检测"""

    @property
    def name(self) -> str:
        return "frame_sample"

    @property
    def dependencies(self) -> list[str]:
        return ["decode"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config

    async def process(self, ctx: PipelineContext) -> None:
        if ctx.raw_frames is None or len(ctx.raw_frames) == 0:
            logger.warning("No frames to sample")
            return

        fps = ctx.fps or 30.0
        total_frames = len(ctx.raw_frames)
        duration = total_frames / fps if fps > 0 else 0

        if duration < self._config.short_video_threshold:
            interval_seconds = self._config.frame_sample_interval_short
        else:
            interval_seconds = self._config.frame_sample_interval_long

        interval_frames = max(1, int(fps * interval_seconds))
        threshold = self._config.scene_change_threshold

        sampled = []
        prev_frame = None
        for i in range(total_frames):
            frame = ctx.raw_frames[i]
            timestamp = i / fps if fps > 0 else float(i)
            is_sample = (i % interval_frames == 0)
            if not is_sample and prev_frame is not None:
                diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)).mean()
                if diff > threshold:
                    is_sample = True
            if is_sample:
                sampled.append(SampledFrame(frame_index=i, timestamp=timestamp, data=frame.copy()))
            prev_frame = frame

        ctx.sampled_frames = sampled if sampled else None
        ctx.stats["sampled_frames"] = float(len(sampled))
        logger.info(f"Sampled {len(sampled)} frames from {total_frames} total")
