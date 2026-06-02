import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set
from uuid import uuid4, UUID


@dataclass
class UploadSession:
    upload_id: UUID
    filename: str
    file_size: int
    chunk_size: int
    total_chunks: int
    status: str = "initiated"


class ChunkedUploadManager:
    def __init__(self, upload_dir: str = "./temp/chunks", redis_client=None,
                 chunk_size: int = 5 * 1024 * 1024, expiry_hours: int = 24):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.redis = redis_client
        self.chunk_size = chunk_size
        self.expiry_hours = expiry_hours

    async def create_upload(self, filename: str, file_size: int,
                            chunk_size: Optional[int] = None) -> UploadSession:
        upload_id = uuid4()
        actual_chunk_size = chunk_size or self.chunk_size
        total_chunks = (file_size + actual_chunk_size - 1) // actual_chunk_size
        session_dir = self.upload_dir / upload_id.hex
        session_dir.mkdir(parents=True, exist_ok=True)
        return UploadSession(upload_id=upload_id, filename=filename,
                             file_size=file_size, chunk_size=actual_chunk_size,
                             total_chunks=total_chunks)

    async def save_chunk(self, upload_id: UUID, chunk_index: int, data: bytes) -> bool:
        session_dir = self.upload_dir / upload_id.hex
        if not session_dir.exists():
            return False
        chunk_path = session_dir / f"chunk_{chunk_index:06d}"
        try:
            chunk_path.write_bytes(data)
        except OSError:
            return False
        if self.redis:
            key = f"chunked_upload:{upload_id}"
            await self.redis.sadd(key, str(chunk_index))
            await self.redis.expire(key, self.expiry_hours * 3600)
        return True

    async def get_uploaded_chunks(self, upload_id: UUID) -> Set[int]:
        if not self.redis:
            return set()
        key = f"chunked_upload:{upload_id}"
        members = await self.redis.smembers(key)
        return {int(m.decode()) if isinstance(m, bytes) else int(m) for m in members}

    async def is_complete(self, upload_id: UUID, total_chunks: int) -> bool:
        uploaded = await self.get_uploaded_chunks(upload_id)
        return len(uploaded) == total_chunks

    def merge_chunks(self, upload_id: UUID, total_chunks: int, output_path: Path) -> bool:
        session_dir = self.upload_dir / upload_id.hex
        if not session_dir.exists():
            return False
        try:
            with open(output_path, "wb") as outfile:
                for i in range(total_chunks):
                    chunk_path = session_dir / f"chunk_{i:06d}"
                    if not chunk_path.exists():
                        return False
                    outfile.write(chunk_path.read_bytes())
            shutil.rmtree(session_dir, ignore_errors=True)
            return True
        except OSError:
            return False

    async def cleanup(self, upload_id: UUID) -> None:
        session_dir = self.upload_dir / upload_id.hex
        shutil.rmtree(session_dir, ignore_errors=True)
        if self.redis:
            await self.redis.delete(f"chunked_upload:{upload_id}")
