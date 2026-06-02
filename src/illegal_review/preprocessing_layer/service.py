import logging
from src.illegal_review.data_models import (
    InputResult, PreprocessingResult, FrameData, AudioData,
)
from src.illegal_review.preprocessing_layer.pipeline import Pipeline
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.preprocessing_layer.stages.decode_stage import DecodeStage
from src.illegal_review.preprocessing_layer.stages.audio_extract_stage import AudioExtractStage
from src.illegal_review.preprocessing_layer.stages.frame_sample_stage import FrameSampleStage
from src.illegal_review.preprocessing_layer.stages.speech_stage import SpeechStage
from src.illegal_review.preprocessing_layer.stages.ocr_stage import OCRStage
from src.illegal_review.preprocessing_layer.utils.temp_cleaner import TempCleaner
from src.illegal_review.preprocessing_layer.utils.frame_io import encode_frame
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class PreprocessingService:
    """预处理层对外门面"""

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        self.pipeline = Pipeline()
        self.pipeline.add_stage(DecodeStage(config))
        self.pipeline.add_stage(AudioExtractStage(config))
        self.pipeline.add_stage(FrameSampleStage(config))
        self.pipeline.add_stage(SpeechStage(config))
        self.pipeline.add_stage(OCRStage(config))

    async def process(self, input_result: InputResult) -> PreprocessingResult:
        ctx = PipelineContext(
            input_id=input_result.input_id,
            video_path=input_result.temp_path,
            config=self._config,
            _input_metadata=input_result.video_metadata,
        )
        async with TempCleaner(ctx):
            ctx = await self.pipeline.run(ctx)
        return self._to_result(ctx)

    def _to_result(self, ctx: PipelineContext) -> PreprocessingResult:
        frames = None
        if ctx.sampled_frames:
            frames = [
                FrameData(
                    frame_index=f.frame_index, timestamp=f.timestamp,
                    image_data=encode_frame(f.data, self._config),
                    width=f.data.shape[1], height=f.data.shape[0],
                )
                for f in ctx.sampled_frames
            ]

        audio = None
        if ctx.audio_path:
            audio = AudioData(
                audio_path=ctx.audio_path,
                sample_rate=self._config.audio_sample_rate,
                duration=ctx.audio_duration or 0, channels=1,
            )

        return PreprocessingResult(
            input_id=ctx.input_id, frames=frames, audio=audio,
            transcript=ctx.transcript, transcript_segments=ctx.transcript_segments,
            ocr_results=ctx.ocr_results, metadata=ctx._input_metadata,
            processing_stats={
                **ctx.stats,
                "total_frames": float(ctx.total_frames or 0),
                "sampled_frames": float(len(ctx.sampled_frames) if ctx.sampled_frames else 0),
                "error_count": float(len(ctx.errors)),
            },
        )
