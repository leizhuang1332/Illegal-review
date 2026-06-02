"""
临时文件管理器

负责临时文件的创建、生命周期管理和清理回收。
支持 TTL 过期清理、活跃任务文件保护、磁盘使用率监控。
"""
import os
import shutil
import time
import threading
from pathlib import Path
from typing import List, Optional, Set
from uuid import uuid4


class TempFileManager:
    """临时文件管理器

    提供临时文件路径生成、活跃状态标记、TTL 过期清理及磁盘用量监控功能。
    使用 threading.Lock 保证活跃文件集合的线程安全。
    """

    def __init__(
        self,
        temp_dir: str = "./temp",
        ttl_hours: int = 24,
        warning_threshold: float = 0.8,
        archive_enabled: bool = False,
        archive_endpoint: str = "",
        archive_bucket: str = "illegal-review-input",
    ):
        """
        Args:
            temp_dir: 临时文件根目录。
            ttl_hours: 文件过期时间（小时）。
            warning_threshold: 磁盘使用率告警阈值 [0.0, 1.0]。
            archive_enabled: 是否启用归档（预留）。
            archive_endpoint: 归档服务端点（预留）。
            archive_bucket: 归档桶名称（预留）。
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.warning_threshold = warning_threshold
        self.archive_enabled = archive_enabled
        self.archive_endpoint = archive_endpoint
        self.archive_bucket = archive_bucket
        self._active_files: Set[str] = set()
        self._lock = threading.Lock()

    def create_temp_path(self, original_filename: str) -> Path:
        """根据原始文件名生成临时文件路径。

        每个文件放在以 uuid 前 8 位命名的子目录下以避免同名冲突。
        文件扩展名从原始文件名继承；若无法识别则默认使用 .mp4。

        Args:
            original_filename: 原始上传文件名。

        Returns:
            生成的临时文件 Path 对象。
        """
        sub_dir = uuid4().hex[:8]
        ext = Path(original_filename).suffix if "." in original_filename else ".mp4"
        dir_path = self.temp_dir / sub_dir
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"input{sub_dir}{ext}"

    def mark_active(self, file_path: Path, task_id: str) -> None:
        """将指定文件标记为活跃（正在被任务使用）。

        被标记的文件在 TTL 清理时将被跳过。

        Args:
            file_path: 文件路径。
            task_id: 关联的任务 ID（用于可追溯性）。
        """
        with self._lock:
            self._active_files.add(str(file_path))

    def mark_completed(self, file_path: Path) -> None:
        """将指定文件标记为已完成，从活跃集合中移除。

        Args:
            file_path: 文件路径。
        """
        with self._lock:
            self._active_files.discard(str(file_path))

    def is_active(self, file_path: Path) -> bool:
        """检查文件是否处于活跃状态。

        Args:
            file_path: 文件路径。

        Returns:
            若文件在活跃集合中返回 True，否则 False。
        """
        return str(file_path) in self._active_files

    def cleanup_ttl(self, max_age_seconds: Optional[int] = None) -> List[str]:
        """清理超过指定时间的过期文件。

        会跳过仍在活跃集合中的文件。
        删除文件后，若父目录为空则一并清理父目录。

        Args:
            max_age_seconds: 文件最大存活秒数。未指定时使用 ttl_hours 换算。

        Returns:
            被删除文件的路径字符串列表。
        """
        if max_age_seconds is None:
            max_age_seconds = self.ttl_hours * 3600
        now = time.time()
        deleted: List[str] = []
        for entry in self.temp_dir.rglob("*"):
            if not entry.is_file() or self.is_active(entry):
                continue
            try:
                file_age = now - entry.stat().st_mtime
                if file_age > max_age_seconds:
                    entry.unlink(missing_ok=True)
                    deleted.append(str(entry))
                    # 清理空父目录
                    parent = entry.parent
                    if parent != self.temp_dir and not any(parent.iterdir()):
                        parent.rmdir()
            except (OSError, PermissionError):
                continue
        return deleted

    def get_disk_usage_ratio(self) -> float:
        """获取临时目录所在磁盘的使用率。

        Returns:
            [0.0, 1.0] 之间的浮点数，表示磁盘使用比例。
            出错时返回 0.0。
        """
        try:
            usage = shutil.disk_usage(self.temp_dir)
            return usage.used / usage.total
        except OSError:
            return 0.0

    def is_disk_warning(self) -> bool:
        """检查磁盘使用率是否超过告警阈值。

        Returns:
            超过阈值返回 True，否则 False。
        """
        return self.get_disk_usage_ratio() >= self.warning_threshold
