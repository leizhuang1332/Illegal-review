# tests/input_layer/test_url_fetcher.py
import pytest
import httpx
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.input_layer.url_fetcher import UrlFetcher


@pytest.fixture
def fetcher():
    return UrlFetcher(timeout=30, max_size=100 * 1024 * 1024)


@pytest.mark.asyncio
async def test_download_success(fetcher, tmp_path):
    """测试成功下载文件。"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "video/mp4", "content-length": "1000"}

    async def _aiter_bytes():
        yield b"hello " * 200  # ~1000 bytes

    mock_response.aiter_bytes = _aiter_bytes

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_instance.stream.return_value = stream_ctx

        output = tmp_path / "test_download.mp4"
        result = await fetcher.download("https://example.com/video.mp4", output)
        assert result.success is True
        assert result.content_type == "video/mp4"
        assert result.file_size > 0
        assert output.exists() is True


@pytest.mark.asyncio
async def test_download_http_error(fetcher, tmp_path):
    """测试 HTTP 非 200 状态码返回错误。"""
    mock_response = AsyncMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_instance.stream.return_value = stream_ctx

        output = tmp_path / "nope.mp4"
        result = await fetcher.download("https://example.com/notfound", output)
        assert result.success is False
        assert "HTTP 404" in result.error


@pytest.mark.asyncio
async def test_download_size_limit_exceeded(tmp_path):
    """测试文件超过大小限制时返回错误并清理临时文件。"""
    small_fetcher = UrlFetcher(timeout=30, max_size=100)  # 仅允许 100 字节

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "video/mp4", "content-length": "10000"}

    async def _aiter_bytes():
        yield b"x" * 200  # 超过 100 字节限制

    mock_response.aiter_bytes = _aiter_bytes

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_instance.stream.return_value = stream_ctx

        output = tmp_path / "test_too_large.mp4"
        result = await small_fetcher.download("https://example.com/large.mp4", output)
        assert result.success is False
        assert "Size limit exceeded" in result.error
        # 临时文件应被清理
        assert output.exists() is False


@pytest.mark.asyncio
async def test_download_timeout(fetcher, tmp_path):
    """测试下载超时返回错误。"""
    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_instance.stream.side_effect = httpx.TimeoutException("timeout")

        output = tmp_path / "test_timeout.mp4"
        result = await fetcher.download("https://example.com/slow.mp4", output)
        assert result.success is False
        assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_download_connection_failed(fetcher, tmp_path):
    """测试连接失败返回错误。"""
    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_instance.stream.side_effect = httpx.ConnectError("connection refused")

        output = tmp_path / "test_conn_fail.mp4"
        result = await fetcher.download("https://example.com/down.mp4", output)
        assert result.success is False
        assert "connection" in result.error.lower()


@pytest.mark.asyncio
async def test_download_general_exception(fetcher, tmp_path):
    """测试未预期的异常被捕获并返回错误。"""
    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_instance.stream.side_effect = RuntimeError("unexpected error")

        output = tmp_path / "test_generic_err.mp4"
        result = await fetcher.download("https://example.com/err.mp4", output)
        assert result.success is False
        assert result.error == "unexpected error"
