import pytest
import numpy as np
from uuid import uuid4
from unittest.mock import patch, MagicMock
from src.illegal_review.preprocessing_layer.stages.ocr_stage import OCRStage
from src.illegal_review.preprocessing_layer.context import PipelineContext, SampledFrame
from src.illegal_review.config.settings import PreprocessingConfig


class TestOCRStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(ocr_languages=["ch_sim", "en"])

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(input_id=uuid4(), video_path="/tmp/test.mp4", config=config)

    def test_name_and_dependencies(self, config):
        with patch("easyocr.Reader", MagicMock()):
            stage = OCRStage(config)
            assert stage.name == "ocr"
            assert stage.dependencies == ["frame_sample"]

    def test_preloads_easyocr_on_init(self, config):
        mock_reader = MagicMock()
        with patch("easyocr.Reader", mock_reader):
            OCRStage(config)
            mock_reader.assert_called_once_with(["ch_sim", "en"])

    @pytest.mark.asyncio
    async def test_ocr_single_frame(self, config, ctx):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        ctx.sampled_frames = [SampledFrame(frame_index=0, timestamp=0.0, data=frame)]
        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(return_value=[
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "检测文字", 0.95)
        ])
        with patch("easyocr.Reader", MagicMock(return_value=mock_reader)):
            stage = OCRStage(config)
            stage._reader = mock_reader
            await stage.process(ctx)
        assert ctx.ocr_results is not None
        assert len(ctx.ocr_results) == 1
        assert ctx.ocr_results[0].text == "检测文字"
        assert ctx.ocr_results[0].confidence == 0.95
        assert ctx.ocr_results[0].frame_index == 0

    @pytest.mark.asyncio
    async def test_no_sampled_frames_skips(self, config, ctx):
        ctx.sampled_frames = None
        with patch("easyocr.Reader", MagicMock()):
            stage = OCRStage(config)
            await stage.process(ctx)
        assert ctx.ocr_results is None

    @pytest.mark.asyncio
    async def test_ocr_failure_graceful(self, config, ctx):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        ctx.sampled_frames = [SampledFrame(frame_index=0, timestamp=0.0, data=frame)]
        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(side_effect=RuntimeError("OCR engine error"))
        with patch("easyocr.Reader", MagicMock(return_value=mock_reader)):
            stage = OCRStage(config)
            stage._reader = mock_reader
            await stage.process(ctx)
        assert ctx.ocr_results is None
        assert len(ctx.errors) > 0
