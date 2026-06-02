import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.preprocessing_layer.service import PreprocessingService
from src.illegal_review.preprocessing_layer.context import PipelineContext, SampledFrame
from src.illegal_review.data_models import InputResult, SourceInfo, VideoMetadata, PreprocessingResult
from src.illegal_review.config.settings import PreprocessingConfig


def make_input_result(video_path="/tmp/test.mp4", has_metadata=True):
    return InputResult(
        input_id=uuid4(), input_type="file",
        source_info=SourceInfo(original_source="test.mp4", file_size=1024, content_type="video/mp4"),
        video_metadata=VideoMetadata(duration=10.0, fps=30.0, width=1920, height=1080, codec="h264") if has_metadata else None,
        temp_path=video_path, status="completed",
        created_at=datetime.now(timezone.utc), processed_at=datetime.now(timezone.utc),
    )


class TestPreprocessingService:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(temp_cleanup_enabled=False)

    @pytest.fixture
    def service(self, config):
        with patch("src.illegal_review.preprocessing_layer.stages.speech_stage.whisper.load_model", MagicMock()), \
             patch("src.illegal_review.preprocessing_layer.stages.ocr_stage.easyocr.Reader", MagicMock()):
            svc = PreprocessingService(config)
            return svc

    @pytest.mark.asyncio
    async def test_process_full_pipeline(self, service):
        input_result = make_input_result()
        mock_ctx = PipelineContext(
            input_id=input_result.input_id, video_path=input_result.temp_path,
            config=service._config, _input_metadata=input_result.video_metadata,
            fps=30.0, total_frames=300,
            audio_path="/tmp/test_audio.wav", audio_duration=10.0,
            sampled_frames=[SampledFrame(frame_index=0, timestamp=0.0, data=np.zeros((480, 640, 3), dtype=np.uint8))],
            transcript="测试转录文本",
            transcript_segments=[{"text": "测试", "start": 0.0, "end": 1.0}],
            ocr_results=[], stats={"decode_duration_ms": 100.0},
        )
        mock_ctx.raw_frames = MagicMock()
        mock_ctx.raw_frames.spill_count = 0

        with patch.object(service.pipeline, "run", AsyncMock(return_value=mock_ctx)):
            result = await service.process(input_result)

        assert isinstance(result, PreprocessingResult)
        assert result.input_id == input_result.input_id
        assert result.transcript == "测试转录文本"
        assert result.frames is not None
        assert len(result.frames) == 1
        assert isinstance(result.frames[0].image_data, bytes)
        assert result.audio is not None
        assert result.audio.audio_path == "/tmp/test_audio.wav"
        assert "decode_duration_ms" in result.processing_stats

    @pytest.mark.asyncio
    async def test_process_no_audio(self, service):
        input_result = make_input_result()
        mock_ctx = PipelineContext(
            input_id=input_result.input_id, video_path=input_result.temp_path,
            config=service._config, _input_metadata=input_result.video_metadata,
            fps=30.0, total_frames=300, audio_path=None,
            sampled_frames=[SampledFrame(frame_index=0, timestamp=0.0, data=np.zeros((480, 640, 3), dtype=np.uint8))],
            stats={},
        )
        mock_ctx.raw_frames = MagicMock()
        mock_ctx.raw_frames.spill_count = 0
        with patch.object(service.pipeline, "run", AsyncMock(return_value=mock_ctx)):
            result = await service.process(input_result)
        assert result.audio is None
        assert result.transcript is None

    @pytest.mark.asyncio
    async def test_process_metadata_passthrough(self, service):
        input_result = make_input_result()
        mock_ctx = PipelineContext(
            input_id=input_result.input_id, video_path=input_result.temp_path,
            config=service._config, _input_metadata=input_result.video_metadata, stats={},
        )
        mock_ctx.raw_frames = MagicMock()
        mock_ctx.raw_frames.spill_count = 0
        with patch.object(service.pipeline, "run", AsyncMock(return_value=mock_ctx)):
            result = await service.process(input_result)
        assert result.metadata is not None
        assert result.metadata.duration == 10.0
        assert result.metadata.width == 1920
