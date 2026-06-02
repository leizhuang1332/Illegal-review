"""
格式识别与校验模块

Magic Number + 扩展名校验，快速验证视频文件格式。
"""
from dataclasses import dataclass
from pathlib import Path


MAGIC_NUMBERS = {
    b"\x00\x00\x00\x1cftyp": "mp4",
    b"\x00\x00\x00\x20ftyp": "mp4",
    b"ftyp": "mp4",
    b"RIFF": "avi",
    b"\x00\x00\x00\x14ftyp": "mov",
    b"\x1a\x45\xdf\xa3": "mkv",
    b"FLV": "flv",
    b"\x1a\x45\xdf\xa6": "webm",
}
MAGIC_MIN_LENGTH = 4


@dataclass
class FormatResult:
    is_valid: bool
    format_name: str = ""
    message: str = ""


def check_magic_number(file_header: bytes) -> FormatResult:
    """通过文件头 Magic Number 识别视频格式。

    Args:
        file_header: 文件头部字节（至少 8 字节）
    Returns:
        FormatResult(is_valid=True, format_name=...) 或 FormatResult(is_valid=False)
    """
    if len(file_header) < MAGIC_MIN_LENGTH:
        return FormatResult(is_valid=False, message="文件头过短，无法识别")

    for signature, fmt in MAGIC_NUMBERS.items():
        if file_header[: len(signature)] == signature:
            return FormatResult(is_valid=True, format_name=fmt)

    return FormatResult(is_valid=False, message="无法识别的文件格式")


def check_extension(filename: str, supported_formats: list[str]) -> FormatResult:
    """通过文件扩展名辅助校验。

    Args:
        filename: 文件名
        supported_formats: 支持的格式列表（小写，无点号）
    Returns:
        FormatResult
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    if not ext:
        return FormatResult(is_valid=False, message="文件名缺少扩展名")
    if ext in supported_formats:
        return FormatResult(is_valid=True, format_name=ext)
    return FormatResult(is_valid=False, message=f"不支持的格式: .{ext}")
