import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KafkaClient:
    def __init__(self, bootstrap_servers: List[str], topic: str,
                 fallback_dir: str = "./temp/kafka_fallback",
                 fallback_max_size: int = 10 * 1024 * 1024 * 1024,
                 fallback_max_files: int = 1000, retry_interval: int = 10):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_max_size = fallback_max_size
        self.fallback_max_files = fallback_max_files
        self.retry_interval = retry_interval
        self.producer = None
        self._fallback_enabled = True

    async def send(self, message: Dict[str, Any]) -> bool:
        if self.producer is None:
            await self._init_producer()
        if self.producer is not None:
            try:
                await self.producer.send(self.topic, value=message)
                await self.producer.flush()
                return True
            except Exception as e:
                logger.warning(f"Kafka send failed: {e}")
        if self._fallback_enabled:
            return await self._write_fallback(message)
        return False

    async def _init_producer(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers, request_timeout_ms=10_000)
            await self.producer.start()
        except Exception as e:
            logger.error(f"Kafka producer init failed: {e}")
            self.producer = None

    async def _write_fallback(self, message: Dict[str, Any]) -> bool:
        if self._fallback_size_exceeded():
            logger.error("Fallback cache full, dropping message")
            return False
        filename = f"{int(time.time() * 1000)}_{message.get('input_id', 'unknown')}.json"
        filepath = self.fallback_dir / filename
        try:
            filepath.write_text(json.dumps(message, ensure_ascii=False, default=str))
            return True
        except OSError as e:
            logger.error(f"Write fallback failed: {e}")
            return False

    def _fallback_size_exceeded(self) -> bool:
        total_size = sum(f.stat().st_size for f in self.fallback_dir.glob("*") if f.is_file())
        total_files = len(list(self.fallback_dir.glob("*")))
        return total_size > self.fallback_max_size or total_files > self.fallback_max_files

    async def retry_fallback_messages(self) -> int:
        if self.producer is None:
            await self._init_producer()
        if self.producer is None:
            return 0
        success_count = 0
        for filepath in sorted(self.fallback_dir.glob("*.json")):
            try:
                message = json.loads(filepath.read_text())
                await self.producer.send(self.topic, value=message)
                filepath.unlink()
                success_count += 1
            except Exception:
                pass
        if success_count:
            await self.producer.flush()
        return success_count

    async def close(self) -> None:
        if self.producer:
            try:
                await self.producer.stop()
            except Exception:
                pass
            self.producer = None
