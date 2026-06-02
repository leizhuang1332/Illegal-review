# tests/input_layer/test_temp_manager.py
import pytest
import tempfile
from pathlib import Path
from src.illegal_review.input_layer.temp_manager import TempFileManager


@pytest.fixture
def manager():
    tmp_dir = tempfile.mkdtemp()
    mgr = TempFileManager(temp_dir=tmp_dir, ttl_hours=24, warning_threshold=0.8)
    yield mgr
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestTempFileManager:
    def test_create_temp_path(self, manager):
        path = manager.create_temp_path("video.mp4")
        assert str(path).endswith(".mp4")

    def test_mark_active_and_is_active(self, manager):
        path = manager.create_temp_path("test.mp4")
        manager.mark_active(path, task_id="task_1")
        assert manager.is_active(path) is True

    def test_mark_completed_removes_active(self, manager):
        path = manager.create_temp_path("test.mp4")
        manager.mark_active(path, task_id="task_1")
        manager.mark_completed(path)
        assert manager.is_active(path) is False

    def test_cleanup_skips_active_files(self, manager):
        path = manager.create_temp_path("active.mp4")
        Path(path).touch()
        manager.mark_active(path, task_id="task_1")
        deleted = [Path(p) for p in manager.cleanup_ttl(max_age_seconds=0)]
        assert path not in deleted
        assert Path(path).exists()

    def test_cleanup_removes_expired_files(self, manager):
        path = manager.create_temp_path("expired.mp4")
        Path(path).touch()
        deleted = [Path(p) for p in manager.cleanup_ttl(max_age_seconds=0)]
        assert path in deleted
        assert Path(path).exists() is False

    def test_disk_usage_ratio(self, manager):
        ratio = manager.get_disk_usage_ratio()
        assert 0.0 <= ratio <= 1.0
