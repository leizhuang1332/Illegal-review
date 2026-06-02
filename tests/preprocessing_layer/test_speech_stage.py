import pytest
from uuid import uuid4
from unittest.mock import patch, MagicMock
from src.illegal_review.preprocessing_layer.stages.speech_stage import SpeechStage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class TestSpeechStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(whisper_model="small")

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(input_id=uuid4(), video_path="/tmp/test.mp4", config=config)

    def test_name_and_dependencies(self, config):
        with patch("whisper.load_model", MagicMock()):
            stage = SpeechStage(config)
            assert stage.name == "speech"
            assert stage.dependencies == ["audio_extract"]

    def test_preloads_whisper_on_init(self, config):
        mock_load = MagicMock()
        with patch("whisper.load_model", mock_load):
            SpeechStage(config)
            mock_load.assert_called_once_with("small")

    @pytest.mark.asyncio
    async def test_transcribe_success(self, config, ctx):
        ctx.audio_path = "/tmp/audio.wav"
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value={
            "text": "测试文本内容", "segments": [{"text": "测试", "start": 0.0, "end": 1.0}]
        })
        with patch("whisper.load_model", MagicMock(return_value=mock_model)):
            stage = SpeechStage(config)
            stage._model = mock_model
            await stage.process(ctx)
        assert ctx.transcript == "测试文本内容"
        assert ctx.transcript_segments is not None
        assert len(ctx.transcript_segments) == 1

    @pytest.mark.asyncio
    async def test_no_audio_skips(self, config, ctx):
        ctx.audio_path = None
        with patch("whisper.load_model", MagicMock()):
            stage = SpeechStage(config)
            await stage.process(ctx)
        assert ctx.transcript is None
        assert ctx.transcript_segments is None

    @pytest.mark.asyncio
    async def test_transcribe_failure_graceful(self, config, ctx):
        ctx.audio_path = "/tmp/audio.wav"
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        with patch("whisper.load_model", MagicMock(return_value=mock_model)):
            stage = SpeechStage(config)
            stage._model = mock_model
            await stage.process(ctx)
        assert ctx.transcript is None
        assert len(ctx.errors) > 0
        assert "CUDA OOM" in ctx.errors[0]
