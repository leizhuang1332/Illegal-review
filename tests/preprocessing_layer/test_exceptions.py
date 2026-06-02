import pytest
from src.illegal_review.preprocessing_layer.exceptions import (
    PreprocessingError,
    DecodeError,
    AudioExtractError,
    RecognitionError,
    PipelineError,
)


class TestPreprocessingExceptions:
    def test_preprocessing_error_base(self):
        """基类异常"""
        with pytest.raises(PreprocessingError, match="test error"):
            raise PreprocessingError("test error")

    def test_decode_error_is_preprocessing_error(self):
        """DecodeError 是 PreprocessingError 的子类"""
        with pytest.raises(PreprocessingError):
            raise DecodeError("decode failed")

    def test_audio_extract_error(self):
        """AudioExtractError 独立捕获"""
        with pytest.raises(AudioExtractError, match="no audio"):
            raise AudioExtractError("no audio")

    def test_recognition_error(self):
        """RecognitionError"""
        err = RecognitionError("whisper failed")
        assert str(err) == "whisper failed"

    def test_pipeline_error(self):
        """PipelineError 用于调度错误"""
        with pytest.raises(PipelineError, match="Circular dependency"):
            raise PipelineError("Circular dependency detected")

    def test_all_inherit_from_preprocessing_error(self):
        """所有异常都是 PreprocessingError 子类"""
        assert issubclass(DecodeError, PreprocessingError)
        assert issubclass(AudioExtractError, PreprocessingError)
        assert issubclass(RecognitionError, PreprocessingError)
        assert issubclass(PipelineError, PreprocessingError)
