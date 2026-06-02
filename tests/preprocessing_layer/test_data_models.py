import pytest
from uuid import uuid4
from src.illegal_review.data_models import (
    PreprocessingResult, FrameData, AudioData, VideoMetadata,
    TranscriptSegment, OCRResult,
)


class TestPreprocessingResultDefaults:
    def test_all_optional_fields_default_to_none(self):
        """所有 Optional 字段默认值为 None"""
        result = PreprocessingResult(input_id=uuid4())
        assert result.frames is None
        assert result.audio is None
        assert result.transcript is None
        assert result.transcript_segments is None
        assert result.ocr_results is None
        assert result.metadata is None

    def test_all_fields_accept_values(self):
        """所有字段可以正常赋值"""
        rid = uuid4()
        result = PreprocessingResult(
            input_id=rid,
            frames=[
                FrameData(
                    frame_index=0, timestamp=0.0,
                    image_data=b"\xff\xd8\xff\xe0", width=1920, height=1080
                )
            ],
            audio=AudioData(
                audio_path="/tmp/test.wav",
                sample_rate=16000, duration=10.0, channels=1
            ),
            transcript="测试文本",
            transcript_segments=[
                TranscriptSegment(text="测试", start=0.0, end=1.0)
            ],
            ocr_results=[
                OCRResult(text="文字", confidence=0.95, bbox=[0, 0, 100, 50], frame_index=0)
            ],
            metadata=VideoMetadata(
                duration=10.0, fps=30.0, width=1920, height=1080, codec="h264"
            ),
            processing_stats={"total_duration_ms": 500.0, "error_count": 0},
        )
        assert result.audio.duration == 10.0
        assert result.frames[0].width == 1920

    def test_model_dump_works(self):
        """model_dump() 正常序列化（含 bytes 字段）"""
        result = PreprocessingResult(
            input_id=uuid4(),
            frames=[
                FrameData(
                    frame_index=0, timestamp=0.0,
                    image_data=b"\xff\xd8\xff\xe0\x00\x10JFIF", width=1920, height=1080
                )
            ],
        )
        dumped = result.model_dump()
        assert dumped["frames"][0]["image_data"] == b"\xff\xd8\xff\xe0\x00\x10JFIF"

    def test_empty_video_all_none(self):
        """无音轨无文字的视频：audio/transcript/ocr_results 全为 None"""
        result = PreprocessingResult(
            input_id=uuid4(),
            metadata=VideoMetadata(
                duration=5.0, fps=25.0, width=640, height=480, codec="h264"
            ),
        )
        assert result.audio is None
        assert result.transcript is None
        assert result.ocr_results is None
        assert result.metadata is not None
