import logging
from pathlib import Path
from src.illegal_review.preprocessing_layer.context import PipelineContext

logger = logging.getLogger(__name__)


class TempCleaner:
    """异步上下文管理器 — 绑定 Pipeline 生命周期，自动清理临时文件"""

    def __init__(self, ctx: PipelineContext, temp_dir: str = ""):
        self._ctx = ctx
        self._temp_dir = Path(temp_dir) if temp_dir else None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._ctx.config.temp_cleanup_enabled:
            return
        self._cleanup()
        return False

    def _cleanup(self) -> None:
        try:
            if self._ctx.raw_frames is not None:
                self._ctx.raw_frames.cleanup()
            if self._ctx.audio_path:
                try:
                    Path(self._ctx.audio_path).unlink(missing_ok=True)
                except OSError:
                    pass
            if self._temp_dir:
                spill_dir = self._temp_dir / "spill"
                if spill_dir.exists():
                    try:
                        for f in spill_dir.glob("*.jpg"):
                            f.unlink(missing_ok=True)
                        spill_dir.rmdir()
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"TempCleaner: cleanup error (non-fatal): {e}")
