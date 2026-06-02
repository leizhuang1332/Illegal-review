import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch
from src.illegal_review.preprocessing_layer.stages.audio_extract_stage import AudioExtractStage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class TestAudioExtractStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(audio_sample_rate=16000)

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(input_id=uuid4(), video_path="/tmp/test.mp4", config=config)

    def test_name_and_dependencies(self, config):
        stage = AudioExtractStage(config)
        assert stage.name == "audio_extract"
        assert stage.dependencies == ["decode"]

    @pytest.mark.asyncio
    async def test_extract_audio_success(self, config, ctx):
        stage = AudioExtractStage(config)
        mock_helper = AsyncMock()
        mock_helper.extract_audio = AsyncMock(return_value="/tmp/audio.wav")
        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)
        assert ctx.audio_path == "/tmp/audio.wav"

    @pytest.mark.asyncio
    async def test_extract_audio_no_track(self, config, ctx):
        stage = AudioExtractStage(config)
        mock_helper = AsyncMock()
        mock_helper.extract_audio = AsyncMock(return_value=None)
        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)
        assert ctx.audio_path is None
