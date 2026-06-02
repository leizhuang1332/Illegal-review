"""
测试元数据提取器（FFprobe）
"""
import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.input_layer.metadata import (
    extract_metadata,
    _parse_r_frame_rate,
    MetadataResult,
)


class TestExtractMetadata:
    def test_successful_extraction(self):
        mock_stdout = '''
        {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920,
                 "height": 1080, "r_frame_rate": "30/1", "bit_rate": "5000000"},
                {"codec_type": "audio", "codec_name": "aac", "bit_rate": "128000"}
            ],
            "format": {"duration": "120.5", "size": "123456789"}
        }
        '''
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_stdout, stderr="", returncode=0
            )
            result = extract_metadata("/tmp/test.mp4")
            assert result.is_valid is True
            assert result.metadata.duration == 120.5
            assert result.metadata.width == 1920
            assert result.metadata.height == 1080
            assert result.metadata.fps == 30.0
            assert result.metadata.codec == "h264"
            assert result.metadata.audio_codec == "aac"
            assert result.metadata.bitrate == 5000000

    def test_ffprobe_timeout_then_retry_then_fail(self):
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.side_effect = TimeoutError("ffprobe timed out")
            result = extract_metadata("/tmp/test.mp4", timeout=5, retries=1)
            assert result.is_valid is False
            assert mock_run.call_count == 2

    def test_ffprobe_first_timeout_then_success(self):
        mock_stdout = '''
        {
            "streams": [
                {"codec_type": "video", "codec_name": "h264",
                 "width": 640, "height": 480, "r_frame_rate": "24/1"}
            ],
            "format": {"duration": "60.0"}
        }
        '''
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.side_effect = [
                TimeoutError("ffprobe timed out"),
                MagicMock(stdout=mock_stdout, stderr="", returncode=0),
            ]
            result = extract_metadata("/tmp/test.mp4", timeout=5, retries=1)
            assert result.is_valid is True
            assert result.metadata.duration == 60.0

    def test_no_video_stream_fails(self):
        mock_stdout = '''
        {
            "streams": [
                {"codec_type": "audio", "codec_name": "aac"}
            ],
            "format": {"duration": "60.0"}
        }
        '''
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_stdout, stderr="", returncode=0
            )
            result = extract_metadata("/tmp/test.mp4")
            assert result.is_valid is False

    def test_ffprobe_not_found(self):
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffprobe not installed")
            result = extract_metadata("/tmp/test.mp4")
            assert result.is_valid is False
            assert "FFprobe not found" in result.message

    def test_ffprobe_returncode_non_zero(self):
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="ffprobe: error", returncode=1)
            result = extract_metadata("/tmp/test.mp4")
            assert result.is_valid is False
            assert result.message == "ffprobe: error"

    def test_ffprobe_json_decode_error(self):
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not json", stderr="", returncode=0)
            result = extract_metadata("/tmp/test.mp4")
            assert result.is_valid is False
            assert result.message == "Parse failed"

    def test_r_frame_rate_null(self):
        mock_stdout = '''
        {
            "streams": [
                {"codec_type": "video", "codec_name": "h264",
                 "width": 1920, "height": 1080, "r_frame_rate": null}
            ],
            "format": {"duration": "60.0"}
        }
        '''
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_stdout, stderr="", returncode=0
            )
            result = extract_metadata("/tmp/test.mp4")
            assert result.is_valid is False
            assert "Invalid metadata" in result.message

    def test_missing_bit_rate(self):
        mock_stdout = '''
        {
            "streams": [
                {"codec_type": "video", "codec_name": "h264",
                 "width": 640, "height": 480, "r_frame_rate": "30/1"}
            ],
            "format": {"duration": "60.0"}
        }
        '''
        with patch("src.illegal_review.input_layer.metadata.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_stdout, stderr="", returncode=0
            )
            result = extract_metadata("/tmp/test.mp4")
            assert result.is_valid is True
            assert result.metadata.bitrate is None


class TestParseRFrameRate:
    def test_normal_fraction(self):
        assert _parse_r_frame_rate("30/1") == 30.0

    def test_non_integer_fraction(self):
        assert _parse_r_frame_rate("2997/100") == 29.97

    def test_plain_number(self):
        assert _parse_r_frame_rate("25") == 25.0

    def test_zero_denominator(self):
        assert _parse_r_frame_rate("30/0") == 0.0

    def test_invalid_fraction(self):
        assert _parse_r_frame_rate("abc/def") == 0.0

    def test_none_input(self):
        assert _parse_r_frame_rate(None) == 0.0
