# 输入层设计文档

> **视频违规审核系统 — 输入层**
> 状态: 已批准 | 日期: 2026-06-02

---

## 1. 架构概览

输入层作为系统的统一入口，接收三种类型的视频输入（文件上传、URL 抓取、直播流接入），同步完成格式校验和元数据提取后返回结果，并通过 Kafka 异步发送消息到下游预处理层。

**部署形态:** 单体应用模块，与系统其他层共享进程和配置。

**处理模式:** 同步处理 — 上传后立即执行校验和元数据提取，同步返回完整结果（含 `video_metadata`）。

---

## 2. 系统架构

```
客户端
  │
  ├─ POST /video/upload  ──┐
  ├─ POST /video/url       ─┤
  ├─ POST /live/start      ─┤
  └─ POST /upload (分片)   ─┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │  Router     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  InputService  │  ← 业务编排（同步）
                    └──┬───┬───┬───┘
                       │   │   │
          ┌────────────┘   │   └────────────┐
          ▼                ▼                ▼
   ┌────────────┐  ┌────────────┐  ┌──────────────┐
   │FormatChecker│  │  FFprobe  │  │TempFileManager│
   │Magic Number │  │  Metadata │  │ TTL保护/清理  │
   │扩展名校验   │  │ 超时重试  │  │ 磁盘预警      │
   └────────────┘  └────────────┘  └──────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
                    ┌──────▼──────┐
                    │  KafkaClient │  ← 异步发送到下游
                    │ (本地缓存回退)│
                    └─────────────┘
```

---

## 3. 模块职责

### 3.1 核心处理链（同步）

| 模块 | 职责 | 关键设计决策 |
|------|------|-------------|
| `format_checker.py` | Magic Number 识别 + 扩展名辅助校验 | Magic Number 优先，扩展名回退 |
| `metadata.py` | FFprobe 提取元数据 | 支持超时重试（默认 1 次重试） |
| `temp_manager.py` | 临时文件全生命周期管理 | TTL 24h 保留，活跃任务保护防止误删 |
| `service.py` | 编排上述模块 | 简化流程（已移除去重和 ack 机制） |

### 3.2 输入接入模块

| 模块 | 职责 |
|------|------|
| `upload_handler.py` | 单文件上传 + 分片上传（类 tus 协议） |
| `url_fetcher.py` | URL 视频流式下载（httpx，支持超时和大小限制） |
| `live_stream.py` | 直播流 FFmpeg segment muxer 录制 |

### 3.3 辅助模块

| 模块 | 职责 |
|------|------|
| `rate_limiter.py` | 令牌桶全局限速（线程安全） |
| `quota_manager.py` | Redis 用户配额（日上传量、日请求数、并发数） |
| `kafka_client.py` | 异步消息发送 + Kafka 不可达时本地文件缓存回退 |
| `monitoring.py` | Prometheus 指标定义（请求数、耗时、文件大小分布等） |

---

## 4. 核心处理流程

### 4.1 文件上传流程

```
1. 客户端 POST /video/upload (multipart/form-data)
2. Router 写入临时文件
3. service.handle_file_upload()
   ├─ 3.1 Magic Number 校验（读取前 16 字节）
   │   └─ 校验失败 → 回退扩展名校验
   │       └─ 仍失败 → status: failed, 返回 400
   ├─ 3.2 文件大小检查
   │   └─ 超限 → status: failed
   ├─ 3.3 标记活跃任务（temp_manager.mark_active）
   ├─ 3.4 FFprobe 提取元数据（默认超时 20s，重试 1 次）
   │   └─ 全部失败 → status: failed
   ├─ 3.5 构建 IngestResult (status: completed)
   ├─ 3.6 异步发送 Kafka 消息（不阻塞响应）
   └─ 3.7 解除活跃任务保护（temp_manager.mark_completed）
4. 同步返回完整结果
   {
     input_id, status: "completed",
     video_metadata: { duration, fps, width, height, codec, ... },
     temp_path, created_at, processed_at
   }
5. 临时文件在 24h 后由 TTL 清理器自动删除
```

### 4.2 URL 抓取流程

```
1. POST /video/url { url, options }
2. UrlFetcher.download() 流式下载到临时路径
   ├─ HTTP 状态码检查
   ├─ Content-Length 大小限制检查
   ├─ 流式写入，边下载边检查大小限制
   └─ 超时/连接失败/IO 异常 → DownloadResult(success=False)
3. 下载成功 → 调用 service.handle_file_upload()（复用相同逻辑）
4. 返回结果
```

### 4.3 直播流接入流程

```
1. POST /live/start { stream_url, protocol, options }
2. StreamRecorder.start() 启动 FFmpeg 子进程
   ├─ -reconnect 参数实现自动断线重连
   ├─ segment muxer 按 chunk_duration（默认 60s）切片
   └─ 进程启动后 500ms 存活检查
3. 返回 streaming 状态
4. POST /live/stop 终止 FFmpeg 进程
```

---

## 5. 数据模型

### 5.1 IngestResult（输入层输出）

```python
class IngestResult(BaseModel):
    input_id: UUID
    input_type: Literal["file", "url", "live"]
    source_info: SourceInfo           # original_source, file_size, content_type
    video_metadata: Optional[VideoMetadata]  # duration, fps, width, height, codec, audio_codec, bitrate
    temp_path: str                    # 本地文件路径
    status: Literal["pending", "processing", "completed", "failed"]
    error: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]
```

### 5.2 API 请求模型

| 模型 | 用途 | 关键字段 |
|------|------|---------|
| `VideoUploadRequest` | 文件上传 | filename, callback_url, options |
| `UrlFetchRequest` | URL 抓取 | url, callback_url, options |
| `LiveStreamRequest` | 直播流 | stream_url, protocol (rtmp\|hls\|http-flv), options |
| `ChunkedUploadCreateRequest` | 分片上传 | filename, file_size, chunk_size, checksum |

### 5.3 API 响应模型

| 模型 | 用途 |
|------|------|
| `TaskStatusResponse` | 任务状态查询 |
| `ChunkedUploadCreateResponse` | 分片上传创建响应（含 upload_urls） |
| `ErrorResponse` | 统一错误响应 |

---

## 6. 配置模型

`InputLayerConfig` 包含以下配置组：

| 组 | 关键参数 |
|------|---------|
| 格式与大小 | supported_formats, max_file_size (5GB) |
| 文件路径 | temp_dir, temp_file_ttl_hours (24h), warning_threshold (80%) |
| 下载 | download_timeout (30s), max_bandwidth (100MB/s) |
| 分片上传 | chunk_size (5MB), concurrent_chunks (3), expiry_hours (24h) |
| 直播流 | buffer_size (10s), reconnect_attempts (5), reconnect_delay (5s) |
| 限速配额 | rate_limit_rps (100), user_concurrent_limit (5), daily_upload_limit (50GB) |
| Kafka 容错 | fallback_dir, fallback_max_size (10GB), fallback_max_files (1000) |
| FFprobe | timeout (20s), retries (1) |

---

## 7. 错误处理策略

| 错误场景 | HTTP 状态码 | 处理方式 |
|---------|------------|---------|
| 格式校验失败 | 400 | 返回 `{error: "无法识别的文件格式"}` |
| FFprobe 超时 | 200 (failed) | 自动重试 1 次，仍失败则 status=failed |
| 文件超过 5GB | 400 | 返回大小限制信息 |
| URL 下载失败 | 502 | 返回具体错误（超时/连接失败/大小超限） |
| Kafka 不可达 | — | 切换到本地缓存，不影响主流程返回 |
| 磁盘空间预警 | — | Prometheus 上报，不阻塞业务 |

---

## 8. 测试策略

- **单元测试**：每个模块独立测试，mock 外部依赖（FFprobe、Redis、httpx、Kafka）
- **Service 测试**：mock 子模块，验证编排逻辑和错误传递
- **Router 测试**：使用 httpx AsyncClient + ASGITransport 测试接口响应
- **目标覆盖率**：核心逻辑路径 100%，异常路径 90%+

---

## 9. 自我审查

### 9.1 占位符检查
- 无 TBD、TODO 或"待实现"类注释
- 所有模块职责明确，无模糊描述

### 9.2 内部一致性
- 配置模型与所有模块的参数名称和默认值一致
- 数据模型（models.py）与 service.py 的字段引用一致
- Magic Number 优先级高于扩展名，与 format_checker 实现一致
- 同步处理模式贯穿整个设计（无异步轮询/回调依赖）

### 9.3 范围检查
- 聚焦于输入层单一职责：接收输入 → 校验 → 提取元数据 → 发送到下游
- 不包含预处理、分析、存储等后续层职责
- 移除去重和 ack 机制（已在计划中明确）

### 9.4 模糊性检查
- "活跃任务保护"定义明确：标记活跃 → TTL 清理跳过 → 完成后解除
- Kafka 发送是"fire-and-forget"模式，不阻塞主流程
- 分片上传使用 Redis set 跟踪已上传分片，合并后清理
