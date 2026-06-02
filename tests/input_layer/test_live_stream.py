import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.input_layer.live_stream import StreamRecorder


class TestStreamRecorder:
    def test_start_recording_launches_ffmpeg(self):
        recorder = StreamRecorder(output_dir="/tmp/live_test")
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc
            result = recorder.start(
                stream_url="rtmp://example.com/live/stream",
                output_path="/tmp/live_test/out.mkv",
                chunk_duration=60,
            )
            assert result is True
            assert recorder.process is not None
            assert recorder.stream_url == "rtmp://example.com/live/stream"

    def test_stop_recording_terminates_process(self):
        recorder = StreamRecorder(output_dir="/tmp/live_test")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        recorder.process = mock_proc
        recorder.stream_url = "rtmp://example.com/live/stream"
        recorder.stop()
        mock_proc.terminate.assert_called_once()

    def test_stop_when_no_process(self):
        recorder = StreamRecorder(output_dir="/tmp/live_test")
        recorder.stop()
