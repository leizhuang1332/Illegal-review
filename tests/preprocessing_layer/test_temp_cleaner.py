import pytest
import tempfile
from pathlib import Path
from uuid import uuid4
from src.illegal_review.preprocessing_layer.utils.temp_cleaner import TempCleaner
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class TestTempCleaner:
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def ctx(self, temp_dir):
        return PipelineContext(
            input_id=uuid4(), video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_files(self, ctx, temp_dir):
        spill_dir = Path(temp_dir) / "spill"
        spill_dir.mkdir()
        (spill_dir / "frame_00000000.jpg").write_bytes(b"fake jpeg")
        assert spill_dir.exists()
        audio_path = Path(temp_dir) / "audio.wav"
        audio_path.write_bytes(b"fake wav")
        ctx.audio_path = str(audio_path)

        cleaner = TempCleaner(ctx, temp_dir=str(temp_dir))
        async with cleaner:
            pass

        assert not audio_path.exists()
        assert not spill_dir.exists() or not list(spill_dir.glob("*.jpg"))

    @pytest.mark.asyncio
    async def test_cleanup_not_called_when_disabled(self, ctx, temp_dir):
        ctx.config.temp_cleanup_enabled = False
        audio_path = Path(temp_dir) / "audio.wav"
        audio_path.write_bytes(b"fake wav")
        ctx.audio_path = str(audio_path)
        cleaner = TempCleaner(ctx, temp_dir=str(temp_dir))
        async with cleaner:
            pass
        assert audio_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_swallows_errors(self, ctx, temp_dir):
        ctx.audio_path = "/nonexistent/path/audio.wav"
        cleaner = TempCleaner(ctx, temp_dir=str(temp_dir))
        async with cleaner:
            pass
