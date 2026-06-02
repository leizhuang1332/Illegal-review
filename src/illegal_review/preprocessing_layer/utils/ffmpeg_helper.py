import asyncio
import json
import logging
from typing import Optional
import numpy as np

from src.illegal_review.data_models import VideoMetadata
from src.illegal_review.preprocessing_layer.exceptions import DecodeError
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore

logger = logging.getLogger(__name__)


class FFmpegHelper:
    """FFmpeg 命令封装 — 子进程安全模式"""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    async def probe_video(self, path: str, timeout: int = 20) -> Optional[VideoMetadata]:
        """ffprobe 获取视频元数据（兜底用）"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffprobe, "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill(); await proc.wait()
                logger.error(f"ffprobe timeout for {path}")
                return None
            finally:
                if proc.returncode is None:
                    proc.kill(); await proc.wait()

            if proc.returncode != 0:
                logger.warning(f"ffprobe failed for {path}")
                return None

            data = json.loads(stdout.decode())
            video_stream = audio_stream = None
            for s in data.get("streams", []):
                if s.get("codec_type") == "video" and video_stream is None:
                    video_stream = s
                elif s.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = s

            if video_stream is None:
                return None

            fps_str = video_stream.get("r_frame_rate", "0/1")
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0

            return VideoMetadata(
                duration=float(data.get("format", {}).get("duration", video_stream.get("duration", 0))),
                fps=fps, width=video_stream.get("width", 0), height=video_stream.get("height", 0),
                codec=video_stream.get("codec_name", "unknown"),
                audio_codec=audio_stream.get("codec_name") if audio_stream else None,
                bitrate=int(data.get("format", {}).get("bit_rate", 0)) or None,
            )
        except Exception as e:
            logger.error(f"ffprobe error for {path}: {e}")
            return None

    async def extract_audio(self, video_path: str, output_path: str, sample_rate: int = 16000, timeout: int = 60) -> Optional[str]:
        """提取音频 → 16kHz 单声道 PCM WAV，无音轨时返回 None"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffmpeg, "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", output_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill(); await proc.wait()
                logger.error(f"Audio extraction timeout for {video_path}")
                return None
            finally:
                if proc.returncode is None:
                    proc.kill(); await proc.wait()

            if proc.returncode != 0:
                err_msg = stderr[-500:].decode(errors="replace")
                if "matches no streams" in err_msg.lower():
                    logger.info(f"No audio stream in {video_path}")
                    return None
                return None
            return output_path
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            return None

    async def decode_frames(self, video_path: str, store: FrameStore,
                            width: int, height: int, fps: float, timeout: Optional[int] = None) -> None:
        """FFmpeg pipe 模式解码视频帧 → FrameStore"""
        frame_size = width * height * 3
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffmpeg, "-y", "-i", video_path,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                while True:
                    try:
                        raw = await asyncio.wait_for(proc.stdout.read(frame_size), timeout=timeout or 300)
                    except asyncio.TimeoutError:
                        proc.kill(); await proc.wait()
                        raise DecodeError(f"Decode timeout after {len(store)} frames")
                    if not raw or len(raw) < frame_size:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                    store.append(frame)
                await proc.wait()
            finally:
                if proc.returncode is None:
                    proc.kill(); await proc.wait()

            if len(store) == 0:
                stderr = await proc.stderr.read() if proc.stderr else b""
                raise DecodeError(f"No frames decoded: {stderr[-500:].decode(errors='replace')}")
        except DecodeError:
            raise
        except Exception as e:
            raise DecodeError(f"Decode failed for {video_path}: {e}")
