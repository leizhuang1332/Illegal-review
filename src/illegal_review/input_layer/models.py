"""
输入层 API 请求/响应模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from uuid import UUID
from datetime import datetime, timezone

from src.illegal_review.data_models import VideoMetadata, SourceInfo, InputResult


class UploadOptions(BaseModel):
    """上传选项"""
    preserve_original: bool = Field(default=True)
    checksum_sha256: Optional[str] = Field(default=None)


class VideoUploadRequest(BaseModel):
    """视频文件上传请求（multipart/form-data）"""
    filename: str = Field(min_length=1)
    callback_url: Optional[str] = Field(default=None)
    options: UploadOptions = Field(default_factory=UploadOptions)


class UrlFetchRequest(BaseModel):
    """URL 视频获取请求"""
    url: str = Field(..., description="视频URL")
    callback_url: Optional[str] = Field(default=None)
    options: UploadOptions = Field(default_factory=UploadOptions)


class LiveStreamRequest(BaseModel):
    """直播流接入请求"""
    stream_url: str = Field(..., description="直播流URL")
    protocol: str = Field(..., pattern="^(rtmp|hls|http-flv)$")
    callback_url: Optional[str] = Field(default=None)
    options: UploadOptions = Field(default_factory=UploadOptions)


class ChunkedUploadCreateRequest(BaseModel):
    """创建分片上传请求"""
    filename: str = Field(min_length=1)
    file_size: int = Field(gt=0)
    chunk_size: int = Field(default=5 * 1024 * 1024, ge=1 * 1024 * 1024, le=50 * 1024 * 1024)
    checksum_sha256: Optional[str] = Field(default=None)


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    input_id: UUID
    status: Literal["pending", "processing", "completed", "failed"]
    progress: int = Field(default=0, ge=0, le=100)
    video_metadata: Optional[VideoMetadata] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ChunkedUploadCreateResponse(BaseModel):
    """创建分片上传响应"""
    upload_id: UUID
    status: str = "initiated"
    chunk_size: int = Field(gt=0)
    total_chunks: int = Field(gt=0)
    upload_urls: List[str]


class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: str
    message: str
    detail: Optional[str] = None
