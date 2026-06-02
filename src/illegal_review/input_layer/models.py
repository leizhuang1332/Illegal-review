"""
输入层 API 请求/响应模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from uuid import UUID
from datetime import datetime, timezone


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


class VideoMetadata(BaseModel):
    """视频元数据"""
    duration: float = Field(gt=0, description="视频时长（秒）")
    fps: float = Field(gt=0, description="帧率")
    width: int = Field(gt=0, description="宽度（像素）")
    height: int = Field(gt=0, description="高度（像素）")
    codec: str = Field(description="视频编码格式")
    audio_codec: Optional[str] = Field(default=None)
    bitrate: Optional[int] = Field(default=None)


class SourceInfo(BaseModel):
    """源信息"""
    original_source: str
    file_size: Optional[int] = Field(default=None)
    content_type: Optional[str] = Field(default=None)


class IngestResult(BaseModel):
    """输入层输出结果"""
    input_id: UUID
    input_type: Literal["file", "url", "live"]
    source_info: SourceInfo
    video_metadata: Optional[VideoMetadata] = None
    temp_path: str
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None


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
