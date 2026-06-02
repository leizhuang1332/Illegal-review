"""
输入层模块

负责统一接收多种类型的视频输入，进行格式校验和预处理，
转换为标准化的内部数据结构。

支持的输入类型：
- 视频文件：本地视频文件上传
- 视频URL：远程视频链接
- 直播流：实时直播内容
- 视频片段：指定时间范围的片段
"""
from src.illegal_review.input_layer.router import router as input_router
from src.illegal_review.input_layer.service import InputService
from src.illegal_review.input_layer.models import IngestResult, VideoMetadata, SourceInfo

__all__ = ["input_router", "InputService", "IngestResult", "VideoMetadata", "SourceInfo"]
