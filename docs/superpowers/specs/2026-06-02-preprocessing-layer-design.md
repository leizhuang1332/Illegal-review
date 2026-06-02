# 预处理层实现设计文档

## 概述

本文档描述了视频违规审核系统中**预处理层**的详细实现设计。预处理层承接输入层标准化后的视频文件，完成**视频解码 → 音频提取 → 帧采样 → 内容识别**全流程，输出结构化特征数据供下游分析引擎消费。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 架构模式 | 流水线引擎型 (DAG Pipeline) | 天然支持异步并行，可扩展性强 |
| 执行方式 | 混合异步 (asyncio + run_in_executor) | CPU 密集型任务放入线程池，I/O 异步 |
| 帧存储策略 | 混合策略 (短视频内存/长视频 spill 文件) | 平衡内存占用与大视频支持 |
| 模型加载 | Stage 初始化时预加载 | 避免首次处理延迟 |
| 异常处理 | 阶段降级 (部分失败不影响其他分支) | 提高系统鲁棒性 |
| 层级合并 | 解码提取层与预处理层合并 | 简化架构，减少中间数据传递 |

## 模块结构

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
    ├── frame_io.py          # 帧存储混合策略 (FrameStore)
    └── temp_cleaner.py      # 临时文件自动清理
```

## 核心接口

### Stage 抽象基类

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

### PipelineContext

```python
@dataclass
class SampledFrame:
    """采样帧"""
    frame_index: int
    timestamp: float       # 秒
    data: np.ndarray       # 图像数据 (H, W, C)

@dataclass
class PipelineContext:
    input_id: UUID
    video_path: str
    config: PreprocessingConfig
    
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

### Pipeline 调度器

```python
class Pipeline:
    def add_stage(self, stage: Stage) -> None: ...
    async def run(self, ctx: PipelineContext) -> PipelineContext: ...
```

调度逻辑：
1. 构建依赖图 → 拓扑排序
2. 每轮选出所有「依赖已就绪」的阶段
3. 用 `asyncio.gather` 并行执行同一轮阶段
4. 失败阶段的下游被跳过，其他分支继续执行

## 阶段详解

### DAG 结构

```
        decode
       ╱      ╲
audio_extract  frame_sample
      │           │
   speech       ocr
      ╲       ╱
   结果聚合 (PipelineContext)
```

轮次 1: `[decode]`
轮次 2: `[audio_extract, frame_sample]` (并行)
轮次 3: `[speech, ocr]` (并行)

### DecodeStage

- **依赖**: 无（根节点）
- **职责**: 使用 FFmpeg + OpenCV 将视频解码为原始帧序列
- **关键实现**: 通过 FrameStore 管理帧存储（混合策略），metadata 由 ffprobe 提取
- **初始化**: 预加载 OpenCV VideoCapture 参数

### AudioExtractStage

- **依赖**: `decode`
- **职责**: 使用 FFmpeg 提取音频轨道并重采样为 16kHz 单声道 PCM
- **关键实现**: subprocess 调用 FFmpeg，无音轨时静默降级
- **初始化**: 无模型加载，轻量

### FrameSampleStage

- **依赖**: `decode`
- **职责**: 基于时间间隔 + 场景变化检测对帧进行智能采样
- **算法**: 
  - 基础策略：<60s 视频每秒1帧，否则每2秒1帧
  - 智能优化：计算相邻帧像素差均值，超过阈值(默认30)保留场景边界帧
- **初始化**: 无模型加载，轻量

### SpeechStage

- **依赖**: `audio_extract`
- **职责**: 调用 Whisper small 模型进行语音转写
- **初始化**: 预加载 Whisper 模型（~1-2s 加载时间）
- **降级**: 音频为空或失败时 transcipt 返回空字符串

### OCRStage

- **依赖**: `frame_sample`
- **职责**: 对采样帧调用 EasyOCR 进行中英文文字识别
- **初始化**: 预加载 EasyOCR 模型
- **输出**: 每帧的 OCR 结果含 text、confidence、bbox、frame_index

## 对外编排 (PreprocessingService)

```python
class PreprocessingService:
    def __init__(self, config: PreprocessingConfig):
        self.pipeline = Pipeline()
        # 注册阶段，初始化时预加载模型
        self.pipeline.add_stage(DecodeStage(config))
        self.pipeline.add_stage(AudioExtractStage(config))
        self.pipeline.add_stage(FrameSampleStage(config))
        self.pipeline.add_stage(SpeechStage(config))
        self.pipeline.add_stage(OCRStage(config))

    async def process(self, input_result: InputResult) -> PreprocessingResult:
        ctx = PipelineContext(
            input_id=input_result.input_id,
            video_path=input_result.temp_path,
            config=self.config,
        )
        async with TempCleaner(ctx):
            ctx = await self.pipeline.run(ctx)
        return self._to_result(ctx)
```

## 工具模块

### FrameStore (frame_io.py)

帧存储混合策略管理器：
- 阈值内帧保持在 `list[np.ndarray]` 内存中
- 超出阈值自动 spill 为磁盘 JPEG 文件，按需懒加载
- 默认阈值 9000 帧 (~5min@30fps)
- 提供 `cleanup()` 方法供 TempCleaner 调用

### FFmpegHelper (ffmpeg_helper.py)

封装三个核心操作：
- `probe_video(path)` — ffprobe 获取元数据
- `extract_audio(video_path, output_path, sr)` — 音频提取
- `decode_frames(video_path, store)` — 逐帧解码写入 FrameStore

### TempCleaner (temp_cleaner.py)

异步上下文管理器，绑定 Pipeline.run 生命周期，自动清理：
- FrameStore spill 文件
- 音频临时 WAV 文件
- 其他中间产物

## 异常体系

```python
class PreprocessingError(Exception):      # 基类
class DecodeError(PreprocessingError):     # 解码失败
class AudioExtractError(PreprocessingError): # 音频提取失败
class RecognitionError(PreprocessingError):  # 识别失败
```

## 测试策略

### 单元测试

| 目标 | 方法 | 关键用例 |
|------|------|---------|
| DecodeStage | 短测试视频 + mock | 正常/文件损坏/格式不兼容 |
| AudioExtractStage | mock FFmpeg | 提取成功/无音轨/损坏 |
| FrameSampleStage | 构造 np 帧序列 | 场景切换/采样间隔/空序列 |
| SpeechStage | mock Whisper 模型 | 正常转写/静音/错误 |
| OCRStage | mock EasyOCR | 中英/空白帧/低置信度 |
| Pipeline | mock 各 Stage | 正常/DAG并行/阶段降级 |

### 集成测试

使用 `tests/fixtures/` 下的真实短测试视频：
- `sample_10s.mp4` — 含语音+文字的正常视频
- `no_audio.mp4` — 无音轨视频
- `corrupted.mp4` — 损坏视频（异常路径）

验证：完整流程执行 → PreprocessingResult 输出 → 临时文件清理

## 输出数据结构

对接上游 (输入层)：

```
InputResult (input_layer) → PreprocessingService → PreprocessingResult
```

输出 PreprocessingResult 供下游分析引擎消费：

| 字段 | 类型 | 下游消费者 |
|------|------|-----------|
| frames | 采样帧列表 | YOLOv8、NSFW、人脸检测 |
| audio | AudioData | 音频特征、敏感语音 |
| transcript | str | BERT语义、敏感词 |
| ocr_results | OCRResult列表 | BERT语义、敏感词 |
| metadata | VideoMetadata | 规则引擎 |
| processing_stats | dict | 监控分析 |
