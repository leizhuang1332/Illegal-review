import pytest
from unittest.mock import AsyncMock, MagicMock, ANY
from datetime import date
from src.illegal_review.input_layer.quota_manager import QuotaManager


@pytest.fixture
def quota_manager():
    return QuotaManager(
        redis_client=MagicMock(),
        daily_upload_limit=1000,
        daily_request_limit=10,
        concurrent_limit=5,
    )


@pytest.mark.asyncio
async def test_check_daily_quota_within_limit(quota_manager):
    quota_manager.redis.get = AsyncMock(return_value="5")
    result = await quota_manager.check_daily_upload("user_1", 100)
    assert result.is_allowed is True


@pytest.mark.asyncio
async def test_check_daily_quota_exceeded(quota_manager):
    quota_manager.redis.get = AsyncMock(return_value="1500")
    result = await quota_manager.check_daily_upload("user_1", 100)
    assert result.is_allowed is False


@pytest.mark.asyncio
async def test_concurrent_limit_within(quota_manager):
    quota_manager.redis.scard = AsyncMock(return_value=3)
    result = await quota_manager.check_concurrent("user_1")
    assert result.is_allowed is True


@pytest.mark.asyncio
async def test_concurrent_limit_exceeded(quota_manager):
    quota_manager.redis.scard = AsyncMock(return_value=5)
    result = await quota_manager.check_concurrent("user_1")
    assert result.is_allowed is False


@pytest.mark.asyncio
async def test_check_daily_requests_within_limit(quota_manager):
    quota_manager.redis.get = AsyncMock(return_value="5")
    result = await quota_manager.check_daily_requests("user_1")
    assert result.is_allowed is True


@pytest.mark.asyncio
async def test_check_daily_requests_exceeded(quota_manager):
    quota_manager.redis.get = AsyncMock(return_value="10")
    result = await quota_manager.check_daily_requests("user_1")
    assert result.is_allowed is False


@pytest.mark.asyncio
async def test_record_upload(quota_manager):
    quota_manager.redis.incrby = AsyncMock(return_value=500)
    quota_manager.redis.expire = AsyncMock(return_value=True)
    await quota_manager.record_upload("user_1", 500)
    quota_manager.redis.incrby.assert_awaited_once_with(
        ANY,
        500,
    )
    quota_manager.redis.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_request(quota_manager):
    quota_manager.redis.incr = AsyncMock(return_value=1)
    quota_manager.redis.expire = AsyncMock(return_value=True)
    await quota_manager.record_request("user_1")
    quota_manager.redis.incr.assert_awaited_once()
    quota_manager.redis.expire.assert_awaited_once()
