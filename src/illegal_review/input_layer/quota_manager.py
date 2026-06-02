from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol


class RedisClient(Protocol):
    async def get(self, key: str) -> Optional[str]: ...
    async def incr(self, key: str) -> int: ...
    async def incrby(self, key: str, amount: int) -> int: ...
    async def expire(self, key: str, seconds: int) -> bool: ...
    async def sadd(self, key: str, member: str) -> int: ...
    async def srem(self, key: str, member: str) -> int: ...
    async def scard(self, key: str) -> int: ...


@dataclass
class QuotaResult:
    is_allowed: bool
    reason: str = ""


class QuotaManager:
    def __init__(self, redis_client: RedisClient,
                 daily_upload_limit: int = 50 * 1024 * 1024 * 1024,
                 daily_request_limit: int = 1000,
                 concurrent_limit: int = 5):
        self.redis = redis_client
        self.daily_upload_limit = daily_upload_limit
        self.daily_request_limit = daily_request_limit
        self.concurrent_limit = concurrent_limit

    def _daily_key(self, user_id: str, prefix: str) -> str:
        return f"quota:{user_id}:{prefix}:{date.today().isoformat()}"

    async def check_daily_upload(self, user_id: str, file_size: int) -> QuotaResult:
        key = self._daily_key(user_id, "upload")
        current = await self.redis.get(key)
        current_bytes = int(current) if current else 0
        if current_bytes + file_size > self.daily_upload_limit:
            return QuotaResult(False, "Daily upload limit exceeded")
        return QuotaResult(True)

    async def check_daily_requests(self, user_id: str) -> QuotaResult:
        key = self._daily_key(user_id, "requests")
        current = await self.redis.get(key)
        current_count = int(current) if current else 0
        if current_count + 1 > self.daily_request_limit:
            return QuotaResult(False, "Daily request limit exceeded")
        return QuotaResult(True)

    async def check_concurrent(self, user_id: str) -> QuotaResult:
        key = f"active_tasks:{user_id}"
        count = await self.redis.scard(key)
        if count >= self.concurrent_limit:
            return QuotaResult(False, "Concurrent limit exceeded")
        return QuotaResult(True)

    async def record_upload(self, user_id: str, file_size: int) -> None:
        key = self._daily_key(user_id, "upload")
        await self.redis.incrby(key, file_size)
        await self.redis.expire(key, 86400)

    async def record_request(self, user_id: str) -> None:
        key = self._daily_key(user_id, "requests")
        await self.redis.incr(key)
        await self.redis.expire(key, 86400)
