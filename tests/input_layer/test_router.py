import pytest
from httpx import AsyncClient, ASGITransport
from src.illegal_review.input_layer.router import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/input/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "input_layer"


@pytest.mark.asyncio
async def test_get_task_status_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/input/tasks/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
