import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from src.illegal_review.input_layer.kafka_client import KafkaClient


@pytest.fixture
def kafka_client(tmp_path):
    client = KafkaClient(
        bootstrap_servers=["localhost:9092"],
        topic="video_tasks",
        fallback_dir=str(tmp_path / "kafka_fallback"),
        retry_interval=5,
    )
    return client


@pytest.mark.asyncio
async def test_send_success(kafka_client):
    kafka_client.producer = AsyncMock()
    kafka_client.producer.send = AsyncMock(return_value=MagicMock())
    kafka_client.producer.flush = AsyncMock()
    result = await kafka_client.send({"input_id": "test", "type": "file"})
    assert result is True


@pytest.mark.asyncio
async def test_send_fallback_on_kafka_failure(kafka_client):
    kafka_client.producer = AsyncMock()
    kafka_client.producer.send = AsyncMock(side_effect=Exception("Kafka down"))
    kafka_client._fallback_enabled = True
    result = await kafka_client.send({"input_id": "test", "type": "file"})
    assert result is True


@pytest.mark.asyncio
async def test_retry_fallback_messages(kafka_client):
    kafka_client.producer = AsyncMock()
    kafka_client.producer.send = AsyncMock(return_value=MagicMock())
    kafka_client.producer.flush = AsyncMock()
    fallback_file = kafka_client.fallback_dir / "test_message.json"
    fallback_file.write_text(json.dumps({"input_id": "retry_test"}))
    await kafka_client.retry_fallback_messages()
    assert fallback_file.exists() is False
