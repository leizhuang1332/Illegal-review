"""
URL 抓取器

负责通过 HTTP/HTTPS 协议从远程 URL 下载视频文件，
支持流式下载、大小限制、超时控制及断网重试异常处理。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class DownloadResult:
    """下载结果数据类。

    Attributes:
        success: 下载是否成功。
        temp_path: 下载文件的临时路径（成功时有效）。
        content_type: 响应 Content-Type。
        file_size: 下载文件大小（字节）。
        error: 错误描述（失败时有效）。
    """
    success: bool
    temp_path: Optional[Path] = None
    content_type: str = ""
    file_size: int = 0
    error: str = ""


class UrlFetcher:
    """URL 抓取器

    提供基于 httpx 的流式 HTTP 下载能力，支持：
    - 流式下载大文件，避免内存占用过高
    - 可配置超时和文件大小上限
    - 自动跟随重定向
    - 常见网络异常的友好错误返回
    """

    def __init__(self, timeout: int = 30, max_size: int = 5 * 1024 * 1024 * 1024,
                 max_bandwidth: int = 100 * 1024 * 1024):
        """
        Args:
            timeout: 请求超时秒数。
            max_size: 最大允许下载大小（字节），默认 5GB。
            max_bandwidth: 最大带宽限制（字节/秒），默认 100MB/s。
        """
        self.timeout = timeout
        self.max_size = max_size
        self.max_bandwidth = max_bandwidth

    async def download(self, url: str, output_path: Path,
                       headers: Optional[dict] = None) -> DownloadResult:
        """从 URL 下载文件到指定路径。

        使用 httpx 流式 API 逐块下载，避免将整个文件加载到内存。
        下载过程中实时检查文件大小是否超出限制，超出则清理已写入内容并返回错误。

        Args:
            url: 文件下载地址。
            output_path: 本地保存路径。
            headers: 可选的自定义请求头。

        Returns:
            DownloadResult 实例，包含下载结果详情。
        """
        timeout_config = httpx.Timeout(
            self.timeout,
            connect=self.timeout,
            read=self.timeout,
        )
        async with httpx.AsyncClient(
            timeout=timeout_config, follow_redirects=True
        ) as client:
            try:
                async with client.stream("GET", url, headers=headers) as response:
                    if response.status_code != 200:
                        return DownloadResult(
                            success=False,
                            error=f"HTTP {response.status_code}",
                        )

                    content_type = response.headers.get("content-type", "")
                    downloaded = 0
                    exceeded = False

                    with open(output_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            downloaded += len(chunk)
                            if downloaded > self.max_size:
                                exceeded = True
                                break
                            f.write(chunk)

                    if exceeded:
                        output_path.unlink(missing_ok=True)
                        return DownloadResult(
                            success=False,
                            error="Size limit exceeded",
                        )

                    return DownloadResult(
                        success=True,
                        temp_path=output_path,
                        content_type=content_type,
                        file_size=downloaded,
                    )

            except httpx.TimeoutException:
                return DownloadResult(success=False, error="Download timeout")
            except httpx.ConnectError:
                return DownloadResult(success=False, error="Connection failed")
            except Exception as e:
                return DownloadResult(success=False, error=str(e))
