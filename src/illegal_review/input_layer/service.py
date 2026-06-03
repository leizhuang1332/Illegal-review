import logging
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timezone

from src.illegal_review.data_models import InputResult, SourceInfo
from src.illegal_review.input_layer.format_checker import check_magic_number, check_extension
from src.illegal_review.input_layer.metadata import extract_metadata
from src.illegal_review.input_layer.temp_manager import TempFileManager

logger = logging.getLogger(__name__)


class InputService:
    def __init__(self, config, temp_manager=None, kafka_client=None):
        self.config = config
        self.temp_manager = temp_manager or TempFileManager(
            temp_dir=config.temp_dir,
            ttl_hours=config.temp_file_ttl_hours,
            warning_threshold=config.temp_dir_warning_threshold,
        )
        self.kafka_client = kafka_client
        self._tasks: Dict[UUID, InputResult] = {}

    async def handle_file_upload(self, file_path: Path, filename: str,
                                 checksum_sha256: Optional[str] = None) -> InputResult:
        input_id = uuid4()
        result = InputResult(
            input_id=input_id, input_type="file",
            source_info=SourceInfo(
                original_source=filename,
                file_size=file_path.stat().st_size if file_path.exists() else None,
            ),
            temp_path=str(file_path),
        )
        # 1. Format check
        try:
            header = file_path.read_bytes()[:16]
            fmt_result = check_magic_number(header)
            if not fmt_result.is_valid:
                ext_result = check_extension(filename, self.config.supported_formats)
                if not ext_result.is_valid:
                    result.status = "failed"
                    result.error = fmt_result.message
                    return result
        except OSError as e:
            result.status = "failed"
            result.error = f"File read failed: {e}"
            return result
        # 2. File size check
        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            result.status = "failed"
            result.error = f"File too large: {file_size}"
            return result
        # 3. Mark active
        self.temp_manager.mark_active(file_path, str(input_id))
        # 4. Metadata extraction
        meta_result = extract_metadata(
            str(file_path), timeout=self.config.ffprobe_timeout,
            retries=self.config.ffprobe_retries)
        if not meta_result.is_valid:
            result.status = "failed"
            result.error = meta_result.message
            self.temp_manager.mark_completed(file_path)
            return result
        result.video_metadata = meta_result.metadata
        result.status = "completed"
        result.processed_at = datetime.now(timezone.utc)
        self._tasks[input_id] = result
        # 5. Kafka
        if self.kafka_client:
            await self._send_kafka_message(result)
        self.temp_manager.mark_completed(file_path)
        return result

    async def _send_kafka_message(self, result: InputResult) -> None:
        if not self.kafka_client:
            return
        message = {
            "message_type": "video_task.created", "version": "1.0",
            "input_id": str(result.input_id), "input_type": result.input_type,
            "temp_path": result.temp_path,
            "video_metadata": result.video_metadata.model_dump() if result.video_metadata else {},
            "source_info": result.source_info.model_dump(),
            "created_at": result.created_at.isoformat(),
        }
        try:
            await self.kafka_client.send(message)
        except Exception as e:
            logger.error(f"Kafka send failed: {e}")

    async def get_task_status(self, input_id: UUID) -> Optional[InputResult]:
        return self._tasks.get(input_id)

    async def cancel_task(self, input_id: UUID) -> bool:
        if input_id in self._tasks:
            result = self._tasks.pop(input_id)
            if result.temp_path:
                Path(result.temp_path).unlink(missing_ok=True)
            return True
        return False
