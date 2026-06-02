import logging
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.preprocessing_layer.utils.ffmpeg_helper import FFmpegHelper
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore
from src.illegal_review.preprocessing_layer.exceptions import DecodeError

logger = logging.getLogger(__name__)


class DecodeStage(Stage):
    """视频解码阶段 — FFmpeg pipe 模式解码 → FrameStore"""

    @property
    def name(self) -> str:
        return "decode"

    @property
    def dependencies(self) -> list[str]:
        return []

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        self._ffmpeg = FFmpegHelper(
            ffmpeg_path=config.ffmpeg_path or "ffmpeg",
            ffprobe_path=config.ffmpeg_path or "ffprobe",
        )

    async def process(self, ctx: PipelineContext) -> None:
        metadata = ctx._input_metadata
        if metadata is None:
            metadata = await self._ffmpeg.probe_video(ctx.video_path)
        if metadata is None:
            raise DecodeError(f"Cannot obtain video metadata for {ctx.video_path}")

        store = FrameStore(
            memory_limit=self._config.frame_store_memory_limit,
            spill_dir=f"{ctx.video_path}_spill",
        )
        await self._ffmpeg.decode_frames(
            ctx.video_path, store,
            width=metadata.width, height=metadata.height, fps=metadata.fps,
        )
        ctx.raw_frames = store
        ctx.fps = metadata.fps
        ctx.total_frames = len(store)
        ctx.stats["total_frames"] = float(ctx.total_frames)
        ctx.stats["spill_count"] = float(store.spill_count)
        logger.info(f"Decoded {ctx.total_frames} frames from {ctx.video_path}")
