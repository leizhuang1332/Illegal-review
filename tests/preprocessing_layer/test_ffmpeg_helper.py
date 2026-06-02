import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.preprocessing_layer.utils.ffmpeg_helper import FFmpegHelper
from src.illegal_review.data_models import VideoMetadata


class TestFFmpegHelper:
    @pytest.fixture
    def helper(self):
        return FFmpegHelper(ffmpeg_path="ffmpeg", ffprobe_path="ffprobe")

    @pytest.mark.asyncio
    async def test_probe_video_returns_metadata(self, helper):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(
            b'{"streams":[{"codec_type":"video","duration":"60.0",'
            b'"r_frame_rate":"30/1","width":1920,"height":1080,'
            b'"codec_name":"h264","bit_rate":"2000000"},'
            b'{"codec_type":"audio","codec_name":"aac"}]}',
            b""
        ))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.probe_video("/tmp/test.mp4")
            assert result.duration == 60.0
            assert result.fps == 30.0
            assert result.width == 1920
            assert result.height == 1080
            assert result.codec == "h264"
            assert result.audio_codec == "aac"

    @pytest.mark.asyncio
    async def test_probe_video_nonzero_return(self, helper):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"file not found"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.probe_video("/tmp/missing.mp4")
            assert result is None

    @pytest.mark.asyncio
    async def test_extract_audio_success(self, helper):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.extract_audio("/tmp/test.mp4", "/tmp/out.wav", 16000)
            assert result == "/tmp/out.wav"

    @pytest.mark.asyncio
    async def test_extract_audio_no_stream_graceful(self, helper):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Stream map 'a' matches no streams"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.extract_audio("/tmp/no_audio.mp4", "/tmp/out.wav", 16000)
            assert result is None

    @pytest.mark.asyncio
    async def test_decode_frames_pipe_mode(self, helper):
        import numpy as np
        from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        raw_data = frame.tobytes()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.read = AsyncMock(side_effect=[raw_data, b""])

        store = FrameStore(memory_limit=100, spill_dir="/tmp/test_spill")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            await helper.decode_frames("/tmp/test.mp4", store, width=64, height=48, fps=30.0)
            assert len(store) == 1

    @pytest.mark.asyncio
    async def test_decode_frames_empty_video(self, helper):
        import numpy as np
        from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore
        from src.illegal_review.preprocessing_layer.exceptions import DecodeError

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Invalid data found"))

        store = FrameStore(memory_limit=100, spill_dir="/tmp/test_spill")
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            with pytest.raises(DecodeError):
                await helper.decode_frames("/tmp/bad.mp4", store, width=1920, height=1080, fps=30.0)
