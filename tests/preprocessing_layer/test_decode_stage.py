import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch
from src.illegal_review.preprocessing_layer.stages.decode_stage import DecodeStage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.data_models import VideoMetadata


class TestDecodeStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(frame_store_memory_limit=100)

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(
            input_id=uuid4(), video_path="/tmp/test.mp4",
            config=config,
            _input_metadata=VideoMetadata(
                duration=10.0, fps=30.0, width=1920, height=1080, codec="h264"
            ),
        )

    def test_name_and_dependencies(self, config):
        stage = DecodeStage(config)
        assert stage.name == "decode"
        assert stage.dependencies == []

    @pytest.mark.asyncio
    async def test_decode_uses_input_metadata(self, config, ctx):
        stage = DecodeStage(config)
        mock_helper = AsyncMock()
        mock_helper.decode_frames = AsyncMock()
        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)
        call_kwargs = mock_helper.decode_frames.call_args
        assert call_kwargs[0][0] == "/tmp/test.mp4"
        assert call_kwargs[1]["width"] == 1920
        assert call_kwargs[1]["height"] == 1080
        assert call_kwargs[1]["fps"] == 30.0
        assert ctx.raw_frames is not None
        assert ctx.fps == 30.0

    @pytest.mark.asyncio
    async def test_decode_falls_back_to_probe(self, config, ctx):
        ctx._input_metadata = None
        stage = DecodeStage(config)
        mock_helper = AsyncMock()
        mock_helper.probe_video = AsyncMock(return_value=VideoMetadata(
            duration=5.0, fps=25.0, width=640, height=480, codec="h264"
        ))
        mock_helper.decode_frames = AsyncMock()
        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)
        mock_helper.probe_video.assert_called_once()
        mock_helper.decode_frames.assert_called_once()

    @pytest.mark.asyncio
    async def test_decode_no_metadata_and_probe_fails(self, config, ctx):
        from src.illegal_review.preprocessing_layer.exceptions import DecodeError
        ctx._input_metadata = None
        stage = DecodeStage(config)
        mock_helper = AsyncMock()
        mock_helper.probe_video = AsyncMock(return_value=None)
        with patch.object(stage, "_ffmpeg", mock_helper):
            with pytest.raises(DecodeError):
                await stage.process(ctx)
