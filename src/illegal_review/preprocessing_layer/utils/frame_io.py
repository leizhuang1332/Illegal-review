from pathlib import Path
import cv2
import numpy as np
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.preprocessing_layer.exceptions import PreprocessingError


def encode_frame(data: np.ndarray, config: PreprocessingConfig) -> bytes:
    """将 np.ndarray 编码为 JPEG bytes"""
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    success, buf = cv2.imencode(
        '.jpg', data,
        [cv2.IMWRITE_JPEG_QUALITY, config.frame_encode_quality]
    )
    if not success:
        raise PreprocessingError("Frame encoding failed")
    return buf.tobytes()


class FrameStore:
    """帧存储混合策略：低于阈值全内存，超出 FIFO spill JPEG 磁盘"""

    def __init__(self, memory_limit: int = 9000, spill_dir: str = ""):
        self._memory: list[np.ndarray] = []
        self._spilled: dict[int, Path] = {}
        self._spill_index_offset: int = 0
        self._limit = memory_limit
        self._spill_dir = Path(spill_dir)
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._count: int = 0
        self.spill_count: int = 0

    def append(self, frame: np.ndarray) -> None:
        self._memory.append(frame)
        self._count += 1
        if len(self._memory) > self._limit:
            self._spill_one()

    def _spill_one(self) -> None:
        frame = self._memory.pop(0)
        global_idx = self._spill_index_offset
        self._spill_index_offset += 1
        spill_path = self._spill_dir / f"frame_{global_idx:08d}.jpg"
        cv2.imwrite(str(spill_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self._spilled[global_idx] = spill_path
        self.spill_count += 1

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self._count
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._count})")
        if idx in self._spilled:
            frame = cv2.imread(str(self._spilled[idx]))
            if frame is None:
                raise PreprocessingError(f"Failed to read spilled frame {idx}")
            return frame
        mem_idx = idx - self._spill_index_offset
        return self._memory[mem_idx]

    def __len__(self) -> int:
        return self._count

    def cleanup(self) -> None:
        for path in self._spilled.values():
            path.unlink(missing_ok=True)
        self._spilled.clear()
        self._memory.clear()
        self._count = 0
        self._spill_index_offset = 0
        self.spill_count = 0
