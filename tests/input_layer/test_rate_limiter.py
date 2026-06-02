import time
import pytest
from src.illegal_review.input_layer.rate_limiter import TokenBucket


class TestTokenBucket:
    def test_allow_within_limit(self):
        bucket = TokenBucket(rate=100, capacity=100)
        assert bucket.consume(1) is True

    def test_block_when_exhausted(self):
        bucket = TokenBucket(rate=10, capacity=10)
        for _ in range(10):
            bucket.consume(1)
        assert bucket.consume(1) is False

    def test_refill_over_time(self):
        bucket = TokenBucket(rate=100, capacity=100)
        bucket.tokens = 0
        bucket.last_refill = time.monotonic() - 0.1
        assert bucket.consume(5) is True

    def test_burst_consumption(self):
        bucket = TokenBucket(rate=10, capacity=100)
        for _ in range(100):
            bucket.consume(1)
        assert bucket.consume(1) is False
