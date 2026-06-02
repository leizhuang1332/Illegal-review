import asyncio
import logging
import whisper
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class SpeechStage(Stage):
    """Whisper 语音转写阶段"""

    @property
    def name(self) -> str:
        return "speech"

    @property
    def dependencies(self) -> list[str]:
        return ["audio_extract"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        logger.info(f"Loading Whisper model: {config.whisper_model}")
        self._model = whisper.load_model(config.whisper_model)
        logger.info(f"Whisper model '{config.whisper_model}' loaded")

    async def process(self, ctx: PipelineContext) -> None:
        if ctx.audio_path is None:
            logger.info("No audio path — skipping speech recognition")
            return
        try:
            result = await asyncio.to_thread(self._model.transcribe, ctx.audio_path)
            ctx.transcript = result.get("text", "")
            ctx.transcript_segments = result.get("segments")
            logger.info(f"Transcription complete: {len(ctx.transcript)} chars")
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            ctx.errors.append(f"SpeechStage error: {e}")
