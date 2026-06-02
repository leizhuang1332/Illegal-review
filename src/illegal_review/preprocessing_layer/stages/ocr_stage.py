import asyncio
import logging
import easyocr
from src.illegal_review.data_models import OCRResult as OCRResultModel
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class OCRStage(Stage):
    """EasyOCR 文字识别阶段"""

    @property
    def name(self) -> str:
        return "ocr"

    @property
    def dependencies(self) -> list[str]:
        return ["frame_sample"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        logger.info(f"Loading EasyOCR: languages={config.ocr_languages}")
        self._reader = easyocr.Reader(config.ocr_languages)
        logger.info("EasyOCR loaded")

    async def process(self, ctx: PipelineContext) -> None:
        if ctx.sampled_frames is None:
            logger.info("No sampled frames — skipping OCR")
            return

        results = []
        try:
            for frame in ctx.sampled_frames:
                detections = await asyncio.to_thread(self._reader.readtext, frame.data)
                for bbox, text, confidence in detections:
                    flat_bbox = [int(v) for pt in bbox for v in pt] if bbox else None
                    results.append(OCRResultModel(
                        text=text, confidence=float(confidence),
                        bbox=flat_bbox, frame_index=frame.frame_index,
                    ))
            ctx.ocr_results = results if results else None
            logger.info(f"OCR complete: {len(results)} text regions found")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            ctx.errors.append(f"OCRStage error: {e}")
