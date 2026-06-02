# tests/preprocessing_layer/test_context.py
import pytest
import numpy as np
from uuid import uuid4
from src.illegal_review.preprocessing_layer.context import SampledFrame, PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class TestSampledFrame:
    def test_create_sampled_frame(self):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = SampledFrame(frame_index=5, timestamp=2.5, data=data)
        assert frame.frame_index == 5
        assert frame.timestamp == 2.5
        assert frame.data.shape == (480, 640, 3)
        assert isinstance(frame.data, np.ndarray)


class TestPipelineContext:
    @pytest.fixture
    def ctx(self):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    def test_initial_defaults(self, ctx):
        """初始状态所有处理字段为 None"""
        assert ctx.raw_frames is None
        assert ctx.fps is None
        assert ctx.total_frames is None
        assert ctx.audio_path is None
        assert ctx.audio_duration is None
        assert ctx.sampled_frames is None
        assert ctx.transcript is None
        assert ctx.transcript_segments is None
        assert ctx.ocr_results is None

    def test_stats_and_errors_default_empty(self, ctx):
        assert ctx.stats == {}
        assert ctx.errors == []

    def test_stats_accumulation(self, ctx):
        ctx.stats["decode_duration_ms"] = 100.0
        ctx.stats["total_frames"] = 1500
        assert ctx.stats["decode_duration_ms"] == 100.0

    def test_errors_accumulation(self, ctx):
        ctx.errors.append("AudioExtractError: no audio stream")
        ctx.errors.append("RecognitionError: OCR timeout")
        assert len(ctx.errors) == 2

    def test_input_metadata_carrying(self, ctx):
        from src.illegal_review.data_models import VideoMetadata
        ctx._input_metadata = VideoMetadata(
            duration=120.0, fps=30.0, width=1920, height=1080, codec="h264"
        )
        assert ctx._input_metadata.duration == 120.0
