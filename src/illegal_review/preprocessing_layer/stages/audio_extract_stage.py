import logging
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.preprocessing_layer.utils.ffmpeg_helper import FFmpegHelper

logger = logging.getLogger(__name__)


class AudioExtractStage(Stage):
    """音频提取阶段 — FFmpeg 提取 → 16kHz 单声道 PCM WAV"""

    @property
    def name(self) -> str:
        return "audio_extract"

    @property
    def dependencies(self) -> list[str]:
        return ["decode"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        self._ffmpeg = FFmpegHelper(ffmpeg_path=config.ffmpeg_path or "ffmpeg")

    async def process(self, ctx: PipelineContext) -> None:
        output_path = f"{ctx.video_path}_audio.wav"
        result = await self._ffmpeg.extract_audio(
            ctx.video_path, output_path, sample_rate=self._config.audio_sample_rate,
        )
        if result is not None:
            ctx.audio_path = result
            logger.info(f"Audio extracted to {result}")
        else:
            logger.info(f"No audio track in {ctx.video_path} — silent degradation")
