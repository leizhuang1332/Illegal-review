# 预处理层实现设计文档

> **视频违规审核系统 — 预处理层**
> 状态: 已批准 | 日期: 2026-06-02 | 审查: 2026-06-02

---

## 1. 概述

预处理层承接输入层标准化后的视频文件，完成**视频解码 → 音频提取 → 帧采样 → 内容识别**全流程，输出结构化特征数据供下游分析引擎消费。

### 1.1 与上下游的契约

```
InputResult (输入层) → PreprocessingService.process() → PreprocessingResult (预处理层)
```

- **上游**：通过 `InputResult` 接收（含 video_metadata，由外部调度器传递）
- **下游**：输出 `PreprocessingResult`，通过 Kafka 消息发送给分析引擎层
- **消息消费**：由外部调度器（Worker/Orchestrator）负责 Kafka 消费，不在本层范围内。预处理层只定义纯函数式契约：`async process(input_result) → PreprocessingResult`

---

## 2. 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 架构模式 | 流水线引擎型 (DAG Pipeline) | 天然支持异步并行，可扩展性强 |
| 执行方式 | 混合异步 (asyncio + run_in_executor) | CPU 密集型任务放入线程池，I/O 异步 |
| 帧存储策略 | 混合策略 (FIFO 内存/超阈值 spill JPEG 文件) | 平衡内存占用与大视频支持 |
| 帧编码格式 | JPEG quality=90 | 体积小(~50KB/帧)，对下游模型精度几乎无损 |
| 模型加载 | Stage 构造时预加载 | 避免首次处理延迟，加载失败在启动时暴露 |
| 异常处理 | 阶段降级 (部分失败不影响其他分支) | 提高系统鲁棒性 |
| 层级合并 | 解码提取层与预处理层合并 | 简化架构，减少中间数据传递 |
| 消息消费 | 外部调度器模式 | PreprocessingService 保持纯粹，不关心消息来源 |

---

## 3. 模块结构

```
preprocessing_layer/
├── __init__.py              # 导出 PreprocessingService
├── pipeline.py              # Pipeline DAG 调度器
├── context.py               # PipelineContext 阶段间共享上下文
├── stage.py                 # Stage 抽象基类
├── service.py               # PreprocessingService 对外门面
├── exceptions.py            # 自定义异常体系
├── stages/
│   ├── __init__.py
│   ├── decode_stage.py      # 视频解码阶段
│   ├── audio_extract_stage.py  # 音频提取阶段
│   ├── frame_sample_stage.py   # 帧采样 + 场景检测阶段
│   ├── speech_stage.py      # Whisper 语音转写阶段
│   └── ocr_stage.py         # EasyOCR 文字识别阶段
└── utils/
    ├── __init__.py
    ├── ffmpeg_helper.py     # FFmpeg 命令封装
    ├── frame_io.py          # 帧存储混合策略 (FrameStore) + 编码工具
    └── temp_cleaner.py      # 临时文件自动清理
```

---

## 4. 核心接口

### 4.1 Stage 抽象基类

```python
class Stage(ABC):
    """所有处理阶段的基类"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dependencies(self) -> list[str]: ...

    @abstractmethod
    async def process(self, ctx: PipelineContext) -> None: ...
```

### 4.2 PipelineContext

```python
@dataclass
class SampledFrame:
    """采样帧 (内部使用 np.ndarray，不对外暴露)"""
    frame_index: int
    timestamp: float       # 秒
    data: np.ndarray       # 图像数据 (H, W, C) — 内部高效处理

@dataclass
class PipelineContext:
    input_id: UUID
    video_path: str
    config: PreprocessingConfig

    # 从 InputResult 携带（透传，不重复 ffprobe）
    _input_metadata: Optional[VideoMetadata] = None

    # 解码输出
    raw_frames: Optional[FrameStore] = None
    fps: Optional[float] = None
    total_frames: Optional[int] = None

    # 音频输出
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None

    # 采样输出
    sampled_frames: Optional[list[SampledFrame]] = None

    # 识别输出
    transcript: Optional[str] = None
    transcript_segments: Optional[list[dict]] = None
    ocr_results: Optional[list[OCRResult]] = None

    # 统计与错误
    stats: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
```

### 4.3 Pipeline 调度器

```python
class Pipeline:
    def add_stage(self, stage: Stage) -> None: ...
    async def run(self, ctx: PipelineContext) -> PipelineContext: ...
```

调度逻辑：

1. 每轮选出所有「依赖已就绪且上游未失败」的阶段
2. 用 `asyncio.gather` 并行执行同一轮阶段
3. 上游失败 → 下游跳过，独立分支继续执行
4. 循环检测：ready 为空但有未完成节点 → 抛出 `PipelineError("Circular dependency")`
5. 每阶段自动计时，写入 `ctx.stats[f"{stage.name}_duration_ms"]`

### 4.4 异常体系

```python
class PreprocessingError(Exception):      # 基类
class DecodeError(PreprocessingError):     # 解码失败
class AudioExtractError(PreprocessingError): # 音频提取失败
class RecognitionError(PreprocessingError):  # 识别失败
class PipelineError(PreprocessingError):     # Pipeline 调度错误（如循环依赖）
```

---

## 5. DAG 结构与阶段详解

### 5.1 DAG 结构

```
        decode
       ╱      ╲
audio_extract  frame_sample
      │           │
   speech       ocr
      ╲       ╱
   结果聚合 (PipelineContext → PreprocessingResult)
```

轮次 1: `[decode]`
轮次 2: `[audio_extract, frame_sample]` (并行)
轮次 3: `[speech, ocr]` (并行)

### 5.2 阶段降级规则

| 失败场景 | frames | audio | transcript | ocr_results | Pipeline 状态 |
|----------|--------|-------|-------------|-------------|--------------|
| decode 失败 | None | None | None | None | 整体失败，抛 DecodeError |
| audio_extract 失败 | ✅ | None | None | ✅ | 部分降级 |
| frame_sample 失败 | None | ✅ | ✅ | None | 部分降级 |
| speech 失败 | ✅ | ✅ | None | ✅ | 部分降级 |
| ocr 失败 | ✅ | ✅ | ✅ | None | 部分降级 |

**三条降级规则**：

1. **依赖传播**：上游失败 → 跳过该分支所有下游（ctx.errors 记录原因）
2. **分支隔离**：独立分支的失败不互相影响（audio_extract 失败不影响 ocr）
3. **根节点特殊**：decode 失败 = Pipeline 整体失败，下游无需执行

**None 语义约定**：
- 所有失败/缺失场景统一返回 `None`（不是空对象、空字符串或空列表）
- `audio=None`：可能是无音轨（正常）或 audio_extract 失败（异常）→ 通过 `ctx.errors` 区分
- `transcript=None`：可能是静音视频（正常）或 speech 失败（异常）
- `ocr_results=None`：可能是纯画面无文字（正常）或 ocr 失败（异常）

### 5.3 DecodeStage

- **依赖**: 无（根节点）
- **职责**: 使用 FFmpeg pipe 模式解码视频为原始帧序列
- **实现**:

  - FFmpeg 输出 rgb24 到 stdout pipe，Python 逐帧 `np.frombuffer()` + `reshape()` 构造 ndarray
  - 不使用 cv2.VideoCapture（它内部调 ffmpeg，行为不可控）
  - 帧写入 FrameStore（FIFO 混合策略）
  - fps/width/height 从 `ctx._input_metadata` 读取（不重复 ffprobe）
- **初始化**: 无模型加载，轻量

### 5.4 AudioExtractStage

- **依赖**: `decode`
- **职责**: 使用 FFmpeg 提取音频轨道并重采样为 16kHz 单声道 PCM WAV
- **实现**: subprocess 调用 FFmpeg，无音轨时静默降级（不抛异常，audio_path 保持 None）
- **命令**: `ffmpeg -i <input> -vn -acodec pcm_s16le -ar 16000 -ac 1 <output>.wav`
- **初始化**: 无模型加载，轻量

### 5.5 FrameSampleStage

- **依赖**: `decode`
- **职责**: 基于时间间隔 + 场景变化检测对帧进行智能采样
- **算法**:
  - 基础策略：<60s 视频每秒1帧，否则每2秒1帧
  - 智能优化：计算相邻帧像素差均值，超过阈值(默认30)保留场景边界帧
- **初始化**: 无模型加载，轻量

### 5.6 SpeechStage

- **依赖**: `audio_extract`
- **职责**: 调用 Whisper small 模型进行语音转写
- **初始化**: 构造时预加载 `whisper.load_model("small")`（~1-2s）
- **推理**: 通过 `asyncio.to_thread()` 放入线程池执行（CPU 密集）
- **降级**: audio_path 为 None 或推理失败时 transcript 为 None

### 5.7 OCRStage

- **依赖**: `frame_sample`
- **职责**: 对采样帧调用 EasyOCR 进行中英文文字识别
- **初始化**: 构造时预加载 EasyOCR reader（~2-3s）
- **推理**: 通过 `asyncio.to_thread()` 放入线程池执行
- **输出**: 每帧的 OCR 结果含 text、confidence、bbox、frame_index
- **降级**: sampled_frames 为 None 或推理失败时 ocr_results 为 None

---

## 6. 对外编排 (PreprocessingService)

```python
class PreprocessingService:
    def __init__(self, config: PreprocessingConfig):
        self.pipeline = Pipeline()
        # 注册阶段（构造时预加载模型）
        self.pipeline.add_stage(DecodeStage(config))
        self.pipeline.add_stage(AudioExtractStage(config))
        self.pipeline.add_stage(FrameSampleStage(config))
        self.pipeline.add_stage(SpeechStage(config))   # ← 构造时加载 Whisper
        self.pipeline.add_stage(OCRStage(config))       # ← 构造时加载 EasyOCR

    async def process(self, input_result: InputResult) -> PreprocessingResult:
        ctx = PipelineContext(
            input_id=input_result.input_id,
            video_path=input_result.temp_path,
            config=self.config,
            _input_metadata=input_result.video_metadata,  # 透传元数据
        )
        async with TempCleaner(ctx):
            ctx = await self.pipeline.run(ctx)
        return self._to_result(ctx)

    def _to_result(self, ctx: PipelineContext) -> PreprocessingResult:
        """边界转换：PipelineContext (np.ndarray) → PreprocessingResult (bytes)"""
        frames = None
        if ctx.sampled_frames:
            frames = [
                FrameData(
                    frame_index=f.frame_index,
                    timestamp=f.timestamp,
                    image_data=encode_frame(f.data, self.config),
                    width=f.data.shape[1],
                    height=f.data.shape[0],
                )
                for f in ctx.sampled_frames
            ]

        audio = None
        if ctx.audio_path:
            audio = AudioData(
                audio_path=ctx.audio_path,
                sample_rate=self.config.audio_sample_rate,
                duration=ctx.audio_duration or 0,
                channels=1,
            )

        return PreprocessingResult(
            input_id=ctx.input_id,
            frames=frames,
            audio=audio,
            transcript=ctx.transcript,
            transcript_segments=ctx.transcript_segments,
            ocr_results=ctx.ocr_results,
            metadata=ctx._input_metadata,
            processing_stats={
                **ctx.stats,
                "total_frames": ctx.total_frames or 0,
                "sampled_frames": len(ctx.sampled_frames) if ctx.sampled_frames else 0,
                "error_count": len(ctx.errors),
            },
        )
```

---

## 7. 数据模型 (修正后)

### 7.1 PreprocessingResult

```python
class PreprocessingResult(BaseModel):
    """预处理层输出结果"""
    input_id: UUID
    frames: Optional[List[FrameData]] = None          # frame_sample 失败 → None
    audio: Optional[AudioData] = None                 # 无音轨或提取失败 → None
    transcript: Optional[str] = None                  # 无语音或识别失败 → None
    transcript_segments: Optional[List[TranscriptSegment]] = None
    ocr_results: Optional[List[OCRResult]] = None     # 无文字或识别失败 → None
    metadata: Optional[VideoMetadata] = None          # 从 InputResult 透传
    processing_stats: Dict[str, float] = Field(default_factory=dict)
```

### 7.2 FrameData (bytes 序列化格式)

```python
class FrameData(BaseModel):
    """帧数据 — JPEG bytes 格式，可序列化"""
    frame_index: int
    timestamp: float
    image_data: bytes       # JPEG 编码 (quality=90)
    width: int
    height: int
```

### 7.3 processing_stats 结构

| 键名 | 含义 | 来源 |
|------|------|------|
| `total_duration_ms` | Pipeline 总耗时 | Pipeline.run() |
| `decode_duration_ms` | 解码耗时 | Pipeline 自动计时 |
| `audio_extract_duration_ms` | 音频提取耗时 | Pipeline 自动计时 |
| `frame_sample_duration_ms` | 帧采样耗时 | Pipeline 自动计时 |
| `speech_duration_ms` | 语音转写耗时 | Pipeline 自动计时 |
| `ocr_duration_ms` | OCR耗时 | Pipeline 自动计时 |
| `total_frames` | 视频总帧数 | ctx.total_frames |
| `sampled_frames` | 采样后帧数 | len(sampled_frames) |
| `spill_count` | FrameStore spill 次数 | FrameStore |
| `error_count` | 失败 Stage 数 | len(ctx.errors) |
| `video_duration_s` | 视频时长（秒） | metadata |

---

## 8. 工具模块

### 8.1 FrameStore (frame_io.py)

帧存储混合策略管理器：

- **策略**: FIFO — 最新帧保留在内存（下游大概率访问），最早帧溢出磁盘
- **内存阈值**: `frame_store_memory_limit`（默认 9000 帧，~5min@30fps）
- **Spill 格式**: JPEG（与输出编码一致，共用 `encode_frame()`）
- **Spill 目录**: 项目 `temp_dir` 子目录，由 TempCleaner 统一管理
- **读回**: 懒加载 `cv2.imread()`，读回后**不放回内存**（避免逐出抖动）
- **清理**: `cleanup()` 方法删除所有 spill 文件

### 8.2 encode_frame() (frame_io.py)

```python
def encode_frame(data: np.ndarray, config: PreprocessingConfig) -> bytes:
    """将 np.ndarray 编码为 JPEG bytes"""
    success, buf = cv2.imencode(
        '.jpg', data,
        [cv2.IMWRITE_JPEG_QUALITY, config.frame_encode_quality]
    )
    if not success:
        raise PreprocessingError("Frame encoding failed")
    return buf.tobytes()
```

### 8.3 FFmpegHelper (ffmpeg_helper.py)

封装三个核心操作，统一子进程安全模式：

**安全措施**：

1. **超时保护**: `asyncio.wait_for()` → 超时则 terminate() → force kill()
2. **资源清理**: finally 块 `proc.stdin.close()` + `proc.wait()` 防僵尸进程
3. **错误映射**: stderr 最后 500 字节放入异常消息
4. **音频降级**: 无音轨时 ffmpeg 返回非零退出码 → 捕获后静默返回（不抛异常）

**三个操作**：

| 方法 | 命令 | 输出 | 超时 |
|------|------|------|------|
| `probe_video(path)` | ffprobe -v quiet -print_format json -show_streams | VideoMetadata | 20s |
| `extract_audio(video, output, sr)` | ffmpeg -i in -vn -acodec pcm_s16le -ar 16000 -ac 1 out.wav | WAV路径 | 60s |
| `decode_frames(video, store)` | ffmpeg -i in -f rawvideo -pix_fmt rgb24 pipe: | 逐帧写入FrameStore | 按时长 |

**probe_video 使用策略**：输入层已调用 ffprobe 并将结果放入 `InputResult.video_metadata`。预处理层优先从 `ctx._input_metadata` 读取，仅在缺失时才独立调用 probe_video 作为兜底。

### 8.4 TempCleaner (temp_cleaner.py)

异步上下文管理器，绑定 Pipeline.run 生命周期，自动清理：
- FrameStore spill 文件
- 音频临时 WAV 文件
- 其他中间产物

---

## 9. 配置模型 (修正后)

### 9.1 PreprocessingConfig (合并后)

```python
@dataclass
class PreprocessingConfig:
    """预处理层配置 — 合并原 DecodeExtractConfig"""

    # --- 解码 ---
    ffmpeg_path: Optional[str] = None           # None = 自动 PATH 查找

    # --- 帧采样 ---
    frame_sample_interval_short: int = 1        # <60s 短视频：每 N 秒 1 帧
    frame_sample_interval_long: int = 2         # ≥60s 长视频：每 N 秒 1 帧
    short_video_threshold: int = 60             # 短视频阈值（秒）
    scene_change_threshold: int = 30            # 场景变化检测像素差均值阈值

    # --- 帧存储 ---
    frame_store_memory_limit: int = 9000        # 内存中最大帧数，超出 spill 磁盘
    frame_encode_format: str = "jpeg"           # 帧编码格式
    frame_encode_quality: int = 90              # JPEG quality (1-100)

    # --- 音频 ---
    audio_sample_rate: int = 16000              # 音频重采样目标
    whisper_model: str = "small"                # Whisper 模型大小

    # --- OCR ---
    ocr_languages: List[str] = field(default_factory=lambda: ["ch_sim", "en"])

    # --- 临时文件 ---
    temp_cleanup_enabled: bool = True           # 处理完成后自动清理
```

### 9.2 连带清理

- **删除** `DecodeExtractConfig`（已合并入 PreprocessingConfig）
- **删除** `SystemConfig.decode_extract` 字段
- **同步** `AudioAnalysisConfig.whisper_model` 默认值 `"base"` → `"small"`

---

## 10. 测试策略

### 10.1 单元测试

| 目标 | 方法 | 关键用例 |
|------|------|---------|
| DecodeStage | 短测试视频 + mock FFmpeg | 正常/文件损坏/格式不兼容 |
| AudioExtractStage | mock FFmpeg subprocess | 提取成功/无音轨(降级)/损坏 |
| FrameSampleStage | 构造 np 帧序列 | 场景切换/采样间隔/空序列 |
| SpeechStage | mock Whisper 模型 | 正常转写/静音(→None)/错误 |
| OCRStage | mock EasyOCR | 中英/空白帧/低置信度 |
| Pipeline | mock 各 Stage | 正常/DAG并行/阶段降级/循环依赖检测 |
| FrameStore | 构造超阈值帧序列 | 内存模式/spill触发/懒加载读回/cleanup |
| Service._to_result() | 构造完整/部分/空 PipelineContext | 全部字段映射/None语义/JPEG编码 |

### 10.2 集成测试

使用 `tests/fixtures/` 下的真实短测试视频：
- `sample_10s.mp4` — 含语音+文字的正常视频
- `no_audio.mp4` — 无音轨视频（验证 audio=None 降级）
- `corrupted.mp4` — 损坏视频（异常路径）

验证：完整流程执行 → PreprocessingResult 输出 → 字段 None 语义正确 → 临时文件清理

---

## 11. 自我审查

### 11.1 内部一致性

- 所有 Optional 字段与阶段降级行为一致
- PreprocessingConfig 覆盖设计文档中所有可配置行为
- 数据模型引用与 data_models.py 一致

### 11.2 范围检查

- 聚焦预处理层：解码 → 采样 → 识别 → 输出结构化数据
- 消息消费由外部调度器负责（明确范围边界）
- Prometheus 指标上报不在本层范围

### 11.3 模糊性检查

- None 语义 vs 空值语义已明确定义
- 阶段降级的 3 条规则覆盖所有故障场景
- 帧编码转换的唯一位置：Service._to_result() + FrameStore spill
