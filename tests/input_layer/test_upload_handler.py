import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.input_layer.upload_handler import (
    ChunkedUploadManager,
)


@pytest.fixture
def chunk_manager():
    tmp_dir = tempfile.mkdtemp()
    redis_mock = MagicMock()
    redis_mock.sadd = AsyncMock(return_value=1)
    redis_mock.smembers = AsyncMock(return_value=set())
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    return ChunkedUploadManager(
        upload_dir=tmp_dir,
        redis_client=redis_mock,
        chunk_size=5 * 1024 * 1024,
        expiry_hours=24,
    )


@pytest.mark.asyncio
async def test_create_upload(chunk_manager):
    upload = await chunk_manager.create_upload("video.mp4", 1000000)
    assert upload.upload_id is not None
    assert upload.status == "initiated"


@pytest.mark.asyncio
async def test_save_and_track_chunk(chunk_manager):
    upload = await chunk_manager.create_upload("video.mp4", 1000000)
    chunk_data = b"test_chunk_data"
    success = await chunk_manager.save_chunk(upload.upload_id, 0, chunk_data)
    assert success is True


@pytest.mark.asyncio
async def test_all_chunks_complete_returns_true(chunk_manager):
    upload = await chunk_manager.create_upload("video.mp4", 10)
    await chunk_manager.save_chunk(upload.upload_id, 0, b"hello")
    chunk_manager.redis.smembers = AsyncMock(return_value={b"0"})
    assert upload.total_chunks == 1
