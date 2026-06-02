# tests/input_layer/test_format_checker.py
import pytest
from src.illegal_review.input_layer.format_checker import (
    check_magic_number,
    check_extension,
    FormatResult,
)


class TestMagicNumber:
    def test_mp4_magic_number(self):
        """MP4 文件头（ftyp box 以 00 00 00 1c 66 74 79 70 开头）"""
        header = bytes([0x00, 0x00, 0x00, 0x1c, 0x66, 0x74, 0x79, 0x70])
        result = check_magic_number(header)
        assert result.is_valid is True
        assert result.format_name == "mp4"

    def test_unknown_magic_number(self):
        """未知格式拒绝"""
        header = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
        result = check_magic_number(header)
        assert result.is_valid is False

    def test_empty_header(self):
        """空文件头拒绝"""
        result = check_magic_number(b"")
        assert result.is_valid is False


class TestExtension:
    def test_supported_extension(self):
        result = check_extension("video.mp4", ["mp4", "avi", "mov"])
        assert result.is_valid is True

    def test_unsupported_extension(self):
        result = check_extension("video.exe", ["mp4", "avi", "mov"])
        assert result.is_valid is False

    def test_no_extension(self):
        result = check_extension("video", ["mp4"])
        assert result.is_valid is False

    def test_uppercase_extension(self):
        result = check_extension("video.MP4", ["mp4", "avi"])
        assert result.is_valid is True
