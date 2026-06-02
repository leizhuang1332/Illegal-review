import shutil
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.illegal_review.input_layer.models import (
    UrlFetchRequest, LiveStreamRequest,
    ChunkedUploadCreateRequest, ChunkedUploadCreateResponse,
    TaskStatusResponse,
)
from src.illegal_review.input_layer.service import InputService

router = APIRouter(prefix="/api/v1/input", tags=["Input Layer"])


def get_service() -> InputService:
    from src.illegal_review.config.settings import config
    service = getattr(get_service, "_instance", None)
    if service is None:
        service = InputService(config=config.input_layer)
        get_service._instance = service
    return service


@router.post("/video/upload", response_model=dict)
async def upload_video(
    video_file: UploadFile = File(...),
    filename: Optional[str] = Form(None),
    service: InputService = Depends(get_service),
):
    orig_name = filename or video_file.filename or "video.mp4"
    temp_path = service.temp_manager.create_temp_path(orig_name)
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
    result = await service.handle_file_upload(file_path=temp_path, filename=orig_name)
    if result.status == "failed":
        return JSONResponse(status_code=400, content={"error": result.error, "input_id": str(result.input_id)})
    return {
        "input_id": str(result.input_id), "status": result.status,
        "video_metadata": result.video_metadata.model_dump() if result.video_metadata else None,
        "temp_path": result.temp_path,
        "created_at": result.created_at.isoformat(),
        "processed_at": result.processed_at.isoformat() if result.processed_at else None,
    }


@router.post("/video/url")
async def fetch_video_url(request: UrlFetchRequest, service: InputService = Depends(get_service)):
    from src.illegal_review.input_layer.url_fetcher import UrlFetcher
    fetcher = UrlFetcher(timeout=service.config.download_timeout, max_size=service.config.max_file_size)
    temp_path = service.temp_manager.create_temp_path("url_video.mp4")
    download_result = await fetcher.download(url=request.url, output_path=temp_path)
    if not download_result.success:
        return JSONResponse(status_code=502, content={"error": download_result.error})
    result = await service.handle_file_upload(
        file_path=temp_path, filename=request.url.split("/")[-1] or "url_video.mp4")
    if result.status == "failed":
        return JSONResponse(status_code=400, content={"error": result.error})
    return {
        "input_id": str(result.input_id), "status": result.status,
        "video_metadata": result.video_metadata.model_dump() if result.video_metadata else None,
        "temp_path": result.temp_path, "created_at": result.created_at.isoformat(),
    }


@router.post("/live/start")
async def start_live_stream(request: LiveStreamRequest, service: InputService = Depends(get_service)):
    from src.illegal_review.input_layer.live_stream import StreamRecorder
    opts = request.options or {}
    chunk_duration = opts.get("chunk_duration", 60)
    output_template = str(service.temp_manager.create_temp_path("live_%Y%m%d_%H%M%S_%s.mp4"))
    recorder = StreamRecorder(output_dir=service.temp_manager.temp_dir)
    started = recorder.start(
        stream_url=request.stream_url, output_path=Path(output_template),
        chunk_duration=chunk_duration,
        reconnect_attempts=opts.get("reconnect_attempts", 5),
        reconnect_delay=opts.get("reconnect_delay", 5),
    )
    if not started:
        raise HTTPException(status_code=502, detail="Live stream connection failed")
    return {"input_id": str(uuid4()), "status": "streaming", "stream_url": request.stream_url, "protocol": request.protocol}


@router.post("/live/stop")
async def stop_live_stream(stream_url: str):
    return {"status": "stopped"}


@router.post("/upload", response_model=ChunkedUploadCreateResponse)
async def create_chunked_upload(request: ChunkedUploadCreateRequest):
    from src.illegal_review.config.settings import config
    from src.illegal_review.input_layer.upload_handler import ChunkedUploadManager
    manager = ChunkedUploadManager(
        upload_dir=config.input_layer.temp_dir + "/chunks",
        chunk_size=config.input_layer.chunk_size,
    )
    session = await manager.create_upload(
        filename=request.filename, file_size=request.file_size, chunk_size=request.chunk_size)
    upload_urls = [f"/api/v1/input/upload/{session.upload_id}/chunk/{i}" for i in range(session.total_chunks)]
    return ChunkedUploadCreateResponse(
        upload_id=session.upload_id, status=session.status,
        chunk_size=session.chunk_size, total_chunks=session.total_chunks,
        upload_urls=upload_urls)


@router.patch("/upload/{upload_id}/chunk/{index}")
async def upload_chunk(upload_id: UUID, index: int):
    return {"status": "uploaded", "upload_id": str(upload_id), "chunk_index": index}


@router.get("/upload/{upload_id}")
async def get_chunked_upload_progress(upload_id: UUID):
    return {"upload_id": str(upload_id), "uploaded_chunks": []}


@router.get("/tasks/{input_id}")
async def get_task_status(input_id: UUID, service: InputService = Depends(get_service)):
    result = await service.get_task_status(input_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        input_id=result.input_id, status=result.status,
        video_metadata=result.video_metadata,
        created_at=result.created_at, updated_at=result.processed_at or result.created_at)


@router.delete("/tasks/{input_id}")
async def cancel_task(input_id: UUID, service: InputService = Depends(get_service)):
    cancelled = await service.cancel_task(input_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "cancelled", "input_id": str(input_id)}


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "input_layer"}
