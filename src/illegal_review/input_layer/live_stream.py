import subprocess
import time
from pathlib import Path
from typing import Optional


class StreamRecorder:
    def __init__(self, output_dir: str = "./temp/live"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process: Optional[subprocess.Popen] = None
        self.stream_url: str = ""

    def start(self, stream_url: str, output_path: Path,
              chunk_duration: int = 60, reconnect_attempts: int = 5,
              reconnect_delay: int = 5) -> bool:
        self.stream_url = stream_url
        cmd = [
            "ffmpeg", "-y", "-reconnect", "1", "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1", "-reconnect_delay_max",
            str(reconnect_delay * reconnect_attempts),
            "-i", stream_url, "-c", "copy", "-f", "segment",
            "-segment_time", str(chunk_duration), "-segment_format", "mp4",
            "-reset_timestamps", "1", "-strftime", "1", str(output_path),
        ]
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(0.5)
            if self.process.poll() is not None:
                self.process = None
                return False
            return True
        except (FileNotFoundError, OSError):
            self.process = None
            return False

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.process = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
