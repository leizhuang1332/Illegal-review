import pytest
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from uuid import uuid4
from src.illegal_review.input_layer.service import InputService


@pytest.fixture
def service():
    config_mock = MagicMock()
    config_mock.temp_dir = "/tmp/test_temp"
    config_mock.max_file_size = 5 * 1024 * 1024 * 1024
    config_mock.ffprobe_timeout = 20
    config_mock.ffprobe_retries = 1
    config_mock.supported_formats = ["mp4", "avi", "mov"]
    return InputService(config=config_mock)


@pytest.mark.asyncio
async def test_handle_file_upload_invalid_format(service):
    with patch("builtins.open", unittest.mock.mock_open(read_data=b"garbage_data")):
        with patch("src.illegal_review.input_layer.format_checker.check_magic_number") as mock_check:
            mock_check.return_value = MagicMock(is_valid=False)
            result = await service.handle_file_upload(
                file_path=Path("/tmp/fake.mp4"),
                filename="fake.mp4",
            )
            assert result.status == "failed"


@pytest.mark.asyncio
async def test_get_task_status_not_found(service):
    result = await service.get_task_status(uuid4())
    assert result is None
