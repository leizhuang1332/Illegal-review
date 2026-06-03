"""
元数据提取器（FFprobe）

使用 FFprobe 提取视频文件的元数据，支持超时重试机制。
"""
import json
import subprocess
from dataclasses import dataclass
from typing import Optional
from pydantic import ValidationError
from src.illegal_review.data_models import VideoMetadata


@dataclass
class MetadataResult:
    """元数据提取结果"""
    is_valid: bool
    metadata: Optional[VideoMetadata] = None
    message: str = ""


def _parse_r_frame_rate(r_frame_rate: str) -> float:
    """将 FFprobe 的 r_frame_rate 字段（如 "30/1"）解析为浮点数"""
    if r_frame_rate is None:
        return 0.0
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 0.0
    return float(r_frame_rate)


def extract_metadata(
    file_path: str,
    timeout: int = 20,
    retries: int = 1,
) -> MetadataResult:
    """
    使用 FFprobe 提取视频文件元数据。

    Args:
        file_path: 视频文件路径
        timeout: 每次 ffprobe 调用的超时秒数
        retries: 超时后的重试次数

    Returns:
        MetadataResult，包含提取结果或错误信息
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", file_path,
    ]
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, TimeoutError):
            last_error = f"FFprobe timeout (attempt {attempt + 1})"
            continue
        except FileNotFoundError:
            return MetadataResult(is_valid=False, message="FFprobe not found")

        if result.returncode != 0:
            return MetadataResult(is_valid=False, message=result.stderr.strip())

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return MetadataResult(is_valid=False, message="Parse failed")

        video_stream = None
        audio_codec = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_codec is None:
                audio_codec = stream.get("codec_name")

        if not video_stream:
            return MetadataResult(is_valid=False, message="No video stream")

        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0))
        bitrate_str = video_stream.get("bit_rate") or fmt.get("bit_rate")

        try:
            metadata = VideoMetadata(
                duration=duration,
                fps=_parse_r_frame_rate(video_stream.get("r_frame_rate", "0/1")),
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                codec=video_stream.get("codec_name", "unknown"),
                audio_codec=audio_codec,
                bitrate=int(bitrate_str) if bitrate_str else None,
            )
        except ValidationError as e:
            return MetadataResult(is_valid=False, message=f"Invalid metadata: {e}")
        return MetadataResult(is_valid=True, metadata=metadata)

    return MetadataResult(is_valid=False, message=f"Extract failed after {retries} retries: {last_error}")
