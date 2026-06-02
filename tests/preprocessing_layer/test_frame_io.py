import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore, encode_frame
from src.illegal_review.config.settings import PreprocessingConfig


class TestEncodeFrame:
    def test_encode_rgb_frame_to_jpeg(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[50:100, 50:100] = [255, 0, 0]
        config = PreprocessingConfig(frame_encode_quality=90)
        result = encode_frame(frame, config)
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:2] == b"\xff\xd8"

    def test_encode_grayscale_frame(self):
        frame = np.zeros((240, 320), dtype=np.uint8)
        config = PreprocessingConfig()
        result = encode_frame(frame, config)
        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"


class TestFrameStore:
    @pytest.fixture
    def spill_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def store(self, spill_dir):
        return FrameStore(memory_limit=5, spill_dir=spill_dir)

    def make_frame(self, value=0):
        return np.full((64, 64, 3), value, dtype=np.uint8)

    def test_initial_empty(self, store):
        assert len(store) == 0

    def test_append_within_limit(self, store):
        for i in range(3):
            store.append(self.make_frame(i))
        assert len(store) == 3
        assert store._count == 3

    def test_append_exceeds_limit_spills(self, store):
        for i in range(7):
            store.append(self.make_frame(i))
        assert len(store) == 7
        assert len(store._spilled) == 2
        for path in store._spilled.values():
            assert os.path.exists(path)

    def test_getitem_from_memory(self, store):
        store.append(self.make_frame(100))
        store.append(self.make_frame(200))
        frame = store[1]
        assert frame[0, 0, 0] == 200

    def test_getitem_from_spill(self, store):
        for i in range(10):
            store.append(self.make_frame(i))
        assert 0 in store._spilled
        frame = store[0]
        assert frame[0, 0, 0] == 0

    def test_cleanup_removes_spill_files(self, store, spill_dir):
        for i in range(10):
            store.append(self.make_frame(i))
        spill_files = list(Path(spill_dir).glob("*.jpg"))
        assert len(spill_files) > 0
        store.cleanup()
        spill_files_after = list(Path(spill_dir).glob("*.jpg"))
        assert len(spill_files_after) == 0
        assert len(store._spilled) == 0
