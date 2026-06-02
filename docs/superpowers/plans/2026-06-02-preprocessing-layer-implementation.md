# 预处理层实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现视频违规审核系统的预处理层，完成视频解码、音频提取、帧采样、语音转写和 OCR 识别的 DAG Pipeline。

**Architecture:** 基于 Stage 抽象基类的 DAG Pipeline，5 个 Stage 按依赖关系分 3 轮并行执行。内部使用 np.ndarray 高效处理，边界统一 JPEG 编码输出。CPU 密集型操作通过 asyncio.to_thread() 放入线程池。

**Tech Stack:** Python 3.10+, asyncio, OpenCV, NumPy, FFmpeg (subprocess), Whisper, EasyOCR, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-06-02-preprocessing-layer-design.md`

---

## 文件结构

```
src/illegal_review/
├── data_models.py                  # [修改] PreprocessingResult 字段 Optional
├── config/settings.py              # [修改] PreprocessingConfig 合并, 删除 DecodeExtractConfig
└── preprocessing_layer/
    ├── __init__.py                 # [修改] 导出 PreprocessingService
    ├── exceptions.py               # [新建]
    ├── context.py                  # [新建]
    ├── stage.py                    # [新建]
    ├── pipeline.py                 # [新建]
    ├── service.py                  # [新建]
    ├── stages/
    │   ├── __init__.py             # [新建]
    │   ├── decode_stage.py         # [新建]
    │   ├── audio_extract_stage.py  # [新建]
    │   ├── frame_sample_stage.py   # [新建]
    │   ├── speech_stage.py         # [新建]
    │   └── ocr_stage.py            # [新建]
    └── utils/
        ├── __init__.py             # [新建]
        ├── ffmpeg_helper.py        # [新建]
        ├── frame_io.py             # [新建]
        └── temp_cleaner.py         # [新建]

tests/
└── preprocessing_layer/
    ├── __init__.py                 # [新建]
    ├── test_exceptions.py          # [新建]
    ├── test_context.py             # [新建]
    ├── test_stage.py               # [新建]
    ├── test_frame_io.py            # [新建]
    ├── test_ffmpeg_helper.py       # [新建]
    ├── test_temp_cleaner.py        # [新建]
    ├── test_pipeline.py            # [新建]
    ├── test_decode_stage.py        # [新建]
    ├── test_audio_extract_stage.py # [新建]
    ├── test_frame_sample_stage.py  # [新建]
    ├── test_speech_stage.py        # [新建]
    ├── test_ocr_stage.py           # [新建]
    └── test_service.py             # [新建]
```

---

### Task 1: 数据模型与配置修正

**Files:**
- Modify: `src/illegal_review/data_models.py`
- Modify: `src/illegal_review/config/settings.py`
- Test: `tests/preprocessing_layer/test_data_models.py` (新建)

- [ ] **Step 1: 编写数据模型验证测试**

```python
# tests/preprocessing_layer/test_data_models.py
import pytest
from uuid import uuid4
from src.illegal_review.data_models import (
    PreprocessingResult, FrameData, AudioData, VideoMetadata,
    TranscriptSegment, OCRResult,
)


class TestPreprocessingResultDefaults:
    def test_all_optional_fields_default_to_none(self):
        """所有 Optional 字段默认值为 None"""
        result = PreprocessingResult(input_id=uuid4())
        assert result.frames is None
        assert result.audio is None
        assert result.transcript is None
        assert result.transcript_segments is None
        assert result.ocr_results is None
        assert result.metadata is None

    def test_all_fields_accept_values(self):
        """所有字段可以正常赋值"""
        import datetime
        rid = uuid4()
        result = PreprocessingResult(
            input_id=rid,
            frames=[
                FrameData(
                    frame_index=0, timestamp=0.0,
                    image_data=b"\xff\xd8\xff\xe0", width=1920, height=1080
                )
            ],
            audio=AudioData(
                audio_path="/tmp/test.wav",
                sample_rate=16000, duration=10.0, channels=1
            ),
            transcript="测试文本",
            transcript_segments=[
                TranscriptSegment(text="测试", start=0.0, end=1.0)
            ],
            ocr_results=[
                OCRResult(text="文字", confidence=0.95, bbox=[0, 0, 100, 50], frame_index=0)
            ],
            metadata=VideoMetadata(
                duration=10.0, fps=30.0, width=1920, height=1080, codec="h264"
            ),
            processing_stats={"total_duration_ms": 500.0, "error_count": 0},
        )
        assert result.audio.duration == 10.0
        assert result.frames[0].width == 1920

    def test_model_dump_works(self):
        """model_dump() 正常序列化（含 bytes 字段）"""
        result = PreprocessingResult(
            input_id=uuid4(),
            frames=[
                FrameData(
                    frame_index=0, timestamp=0.0,
                    image_data=b"\xff\xd8\xff\xe0\x00\x10JFIF", width=1920, height=1080
                )
            ],
        )
        dumped = result.model_dump()
        assert dumped["frames"][0]["image_data"] == b"\xff\xd8\xff\xe0\x00\x10JFIF"

    def test_empty_video_all_none(self):
        """无音轨无文字的视频：audio/transcript/ocr_results 全为 None"""
        result = PreprocessingResult(
            input_id=uuid4(),
            metadata=VideoMetadata(
                duration=5.0, fps=25.0, width=640, height=480, codec="h264"
            ),
        )
        assert result.audio is None
        assert result.transcript is None
        assert result.ocr_results is None
        assert result.metadata is not None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_data_models.py -v
```

Expected: FAIL — `PreprocessingResult` 字段仍是必填

- [ ] **Step 3: 修改 data_models.py — PreprocessingResult 字段 Optional**

```python
# 修改 PreprocessingResult（在原有基础上将 5 个字段改为 Optional，默认 None）
class PreprocessingResult(BaseModel):
    """预处理层输出结果"""
    input_id: UUID = Field(description="追踪ID")
    frames: Optional[List[FrameData]] = Field(default=None, description="采样后的帧序列")
    audio: Optional[AudioData] = Field(default=None, description="音频数据")
    transcript: Optional[str] = Field(default=None, description="完整转录文本")
    transcript_segments: Optional[List[TranscriptSegment]] = Field(default=None, description="分段转录结果")
    ocr_results: Optional[List[OCRResult]] = Field(default=None, description="OCR识别结果")
    metadata: Optional[VideoMetadata] = Field(default=None, description="处理元数据")
    processing_stats: Dict[str, float] = Field(default_factory=dict, description="处理统计信息")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_data_models.py -v
```

Expected: PASS (4/4)

- [ ] **Step 5: 修改 config/settings.py — 合并 PreprocessingConfig**

```python
# 替换 PreprocessingConfig 为合并版本
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


# 删除 DecodeExtractConfig 类

# 修改 AudioAnalysisConfig.whisper_model 默认值
@dataclass
class AudioAnalysisConfig:
    whisper_model: str = "small"  # "base" → "small"
    # ... 其余字段不变


# 修改 SystemConfig — 删除 decode_extract 字段
@dataclass
class SystemConfig:
    input_layer: InputLayerConfig = field(default_factory=InputLayerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    # 删除: decode_extract: DecodeExtractConfig = field(default_factory=DecodeExtractConfig)
    image_recognition: ImageRecognitionConfig = field(default_factory=ImageRecognitionConfig)
    # ... 其余字段不变
```

- [ ] **Step 6: 验证配置导入正常**

```bash
python -c "from src.illegal_review.config.settings import SystemConfig; c = SystemConfig(); print(c.preprocessing.whisper_model, c.preprocessing.frame_store_memory_limit)"
```

Expected: `small 9000`

- [ ] **Step 7: Commit**

```bash
git add src/illegal_review/data_models.py src/illegal_review/config/settings.py tests/preprocessing_layer/test_data_models.py
git commit -m "fix: make PreprocessingResult fields Optional, merge PreprocessingConfig, sync whisper model to small"
```

---

### Task 2: 异常体系

**Files:**
- Create: `src/illegal_review/preprocessing_layer/exceptions.py`
- Create: `tests/preprocessing_layer/test_exceptions.py`

- [ ] **Step 1: 编写异常测试**

```python
# tests/preprocessing_layer/test_exceptions.py
import pytest
from src.illegal_review.preprocessing_layer.exceptions import (
    PreprocessingError,
    DecodeError,
    AudioExtractError,
    RecognitionError,
    PipelineError,
)


class TestPreprocessingExceptions:
    def test_preprocessing_error_base(self):
        """基类异常"""
        with pytest.raises(PreprocessingError, match="test error"):
            raise PreprocessingError("test error")

    def test_decode_error_is_preprocessing_error(self):
        """DecodeError 是 PreprocessingError 的子类"""
        with pytest.raises(PreprocessingError):
            raise DecodeError("decode failed")

    def test_audio_extract_error(self):
        """AudioExtractError 独立捕获"""
        with pytest.raises(AudioExtractError, match="no audio"):
            raise AudioExtractError("no audio")

    def test_recognition_error(self):
        """RecognitionError"""
        err = RecognitionError("whisper failed")
        assert str(err) == "whisper failed"

    def test_pipeline_error(self):
        """PipelineError 用于调度错误"""
        with pytest.raises(PipelineError, match="Circular dependency"):
            raise PipelineError("Circular dependency detected")

    def test_all_inherit_from_preprocessing_error(self):
        """所有异常都是 PreprocessingError 子类"""
        assert issubclass(DecodeError, PreprocessingError)
        assert issubclass(AudioExtractError, PreprocessingError)
        assert issubclass(RecognitionError, PreprocessingError)
        assert issubclass(PipelineError, PreprocessingError)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_exceptions.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现异常类**

```python
# src/illegal_review/preprocessing_layer/exceptions.py
class PreprocessingError(Exception):
    """预处理层基础异常"""
    pass


class DecodeError(PreprocessingError):
    """视频解码失败"""
    pass


class AudioExtractError(PreprocessingError):
    """音频提取失败"""
    pass


class RecognitionError(PreprocessingError):
    """内容识别失败（语音/OCR）"""
    pass


class PipelineError(PreprocessingError):
    """Pipeline 调度错误（如循环依赖、配置错误）"""
    pass
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_exceptions.py -v
```

Expected: PASS (6/6)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/exceptions.py tests/preprocessing_layer/test_exceptions.py
git commit -m "feat: add preprocessing layer exception hierarchy"
```

---

### Task 3: PipelineContext

**Files:**
- Create: `src/illegal_review/preprocessing_layer/context.py`
- Create: `tests/preprocessing_layer/test_context.py`

- [ ] **Step 1: 编写 Context 测试**

```python
# tests/preprocessing_layer/test_context.py
import pytest
import numpy as np
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Optional
from collections.abc import MutableSequence


class TestSampledFrame:
    def test_create_sampled_frame(self):
        from src.illegal_review.preprocessing_layer.context import SampledFrame

        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = SampledFrame(frame_index=5, timestamp=2.5, data=data)
        assert frame.frame_index == 5
        assert frame.timestamp == 2.5
        assert frame.data.shape == (480, 640, 3)
        assert isinstance(frame.data, np.ndarray)


class TestPipelineContext:
    @pytest.fixture
    def ctx(self):
        from src.illegal_review.preprocessing_layer.context import PipelineContext
        from src.illegal_review.config.settings import PreprocessingConfig

        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    def test_initial_defaults(self, ctx):
        """初始状态所有处理字段为 None"""
        assert ctx.raw_frames is None
        assert ctx.fps is None
        assert ctx.total_frames is None
        assert ctx.audio_path is None
        assert ctx.audio_duration is None
        assert ctx.sampled_frames is None
        assert ctx.transcript is None
        assert ctx.transcript_segments is None
        assert ctx.ocr_results is None

    def test_stats_and_errors_default_empty(self, ctx):
        """stats 和 errors 默认为空"""
        assert ctx.stats == {}
        assert ctx.errors == []

    def test_stats_accumulation(self, ctx):
        """stats 可以累加"""
        ctx.stats["decode_duration_ms"] = 100.0
        ctx.stats["total_frames"] = 1500
        assert ctx.stats["decode_duration_ms"] == 100.0

    def test_errors_accumulation(self, ctx):
        """errors 可以累加"""
        ctx.errors.append("AudioExtractError: no audio stream")
        ctx.errors.append("RecognitionError: OCR timeout")
        assert len(ctx.errors) == 2

    def test_input_metadata_carrying(self, ctx):
        """_input_metadata 用于透传输入层元数据"""
        from src.illegal_review.data_models import VideoMetadata

        ctx._input_metadata = VideoMetadata(
            duration=120.0, fps=30.0, width=1920, height=1080, codec="h264"
        )
        assert ctx._input_metadata.duration == 120.0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_context.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 PipelineContext 和 SampledFrame**

```python
# src/illegal_review/preprocessing_layer/context.py
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import numpy as np

from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.data_models import VideoMetadata, OCRResult


@dataclass
class SampledFrame:
    """采样帧（内部使用 np.ndarray，不对外暴露）"""
    frame_index: int
    timestamp: float       # 秒
    data: np.ndarray       # 图像数据 (H, W, C)


@dataclass
class PipelineContext:
    """Pipeline 阶段间共享上下文"""
    input_id: UUID
    video_path: str
    config: PreprocessingConfig

    # 从 InputResult 携带（透传，不重复 ffprobe）
    _input_metadata: Optional[VideoMetadata] = None

    # 解码输出
    raw_frames: Optional["FrameStore"] = None
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

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_context.py -v
```

Expected: PASS (6/6)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/context.py tests/preprocessing_layer/test_context.py
git commit -m "feat: add PipelineContext and SampledFrame"
```

---

### Task 4: Stage 抽象基类

**Files:**
- Create: `src/illegal_review/preprocessing_layer/stage.py`
- Create: `tests/preprocessing_layer/test_stage.py`

- [ ] **Step 1: 编写 Stage 基类测试**

```python
# tests/preprocessing_layer/test_stage.py
import pytest
from abc import abstractmethod
from uuid import uuid4
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class _ConcreteStage(Stage):
    """测试用具体实现"""
    @property
    def name(self) -> str:
        return "test_stage"

    @property
    def dependencies(self) -> list[str]:
        return ["upstream_stage"]

    async def process(self, ctx: PipelineContext) -> None:
        ctx.stats["test_duration_ms"] = 42.0


class TestStageAbstract:
    @pytest.fixture
    def ctx(self):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    def test_cannot_instantiate_abstract(self):
        """不能直接实例化 Stage"""
        with pytest.raises(TypeError):
            Stage()

    def test_concrete_stage_has_name(self):
        stage = _ConcreteStage()
        assert stage.name == "test_stage"

    def test_concrete_stage_has_dependencies(self):
        stage = _ConcreteStage()
        assert stage.dependencies == ["upstream_stage"]

    @pytest.mark.asyncio
    async def test_concrete_stage_process(self, ctx):
        stage = _ConcreteStage()
        await stage.process(ctx)
        assert ctx.stats["test_duration_ms"] == 42.0

    def test_missing_name_raises(self):
        """缺少 name 属性无法实例化"""
        class BadStage(Stage):
            dependencies = []
            async def process(self, ctx): pass

        with pytest.raises(TypeError):
            BadStage()

    def test_missing_dependencies_raises(self):
        """缺少 dependencies 属性无法实例化"""
        class BadStage(Stage):
            name = "bad"
            async def process(self, ctx): pass

        with pytest.raises(TypeError):
            BadStage()
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_stage.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 Stage 抽象基类**

```python
# src/illegal_review/preprocessing_layer/stage.py
from abc import ABC, abstractmethod
from src.illegal_review.preprocessing_layer.context import PipelineContext


class Stage(ABC):
    """所有处理阶段的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """阶段名称，用作 DAG 节点标识"""
        ...

    @property
    @abstractmethod
    def dependencies(self) -> list[str]:
        """依赖的阶段名称列表"""
        ...

    @abstractmethod
    async def process(self, ctx: PipelineContext) -> None:
        """执行阶段处理逻辑，结果写入 ctx"""
        ...
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_stage.py -v
```

Expected: PASS (6/6)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/stage.py tests/preprocessing_layer/test_stage.py
git commit -m "feat: add Stage abstract base class"
```

---

### Task 5: FrameStore 与 encode_frame

**Files:**
- Create: `src/illegal_review/preprocessing_layer/utils/__init__.py` (空文件)
- Create: `src/illegal_review/preprocessing_layer/utils/frame_io.py`
- Create: `tests/preprocessing_layer/test_frame_io.py`

- [ ] **Step 1: 编写 FrameStore 和 encode_frame 测试**

```python
# tests/preprocessing_layer/test_frame_io.py
import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore, encode_frame
from src.illegal_review.config.settings import PreprocessingConfig


class TestEncodeFrame:
    def test_encode_rgb_frame_to_jpeg(self):
        """RGB 帧编码为 JPEG bytes"""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # 画一个红色方块
        frame[50:100, 50:100] = [255, 0, 0]
        config = PreprocessingConfig(frame_encode_quality=90)
        result = encode_frame(frame, config)
        assert isinstance(result, bytes)
        assert len(result) > 0
        # JPEG 以 FF D8 开头
        assert result[:2] == b"\xff\xd8"

    def test_encode_grayscale_frame(self):
        """灰度帧自动转 RGB 后编码"""
        frame = np.zeros((240, 320), dtype=np.uint8)
        config = PreprocessingConfig()
        result = encode_frame(frame, config)
        assert isinstance(result, bytes)
        assert result[:2] == b"\xff\xd8"


class TestFrameStore:
    @pytest.fixture
    def spill_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def store(self, spill_dir):
        return FrameStore(memory_limit=5, spill_dir=spill_dir)

    def make_frame(self, value=0):
        return np.full((64, 64, 3), value, dtype=np.uint8)

    def test_initial_empty(self, store):
        assert len(store) == 0

    def test_append_within_limit(self, store):
        """低于阈值时全部在内存"""
        for i in range(3):
            store.append(self.make_frame(i))
        assert len(store) == 3
        assert store._count == 3

    def test_append_exceeds_limit_spills(self, store):
        """超出阈值时最早帧 spill 磁盘"""
        for i in range(7):
            store.append(self.make_frame(i))
        assert len(store) == 7
        # 应该有 2 帧被 spill
        assert len(store._spilled) == 2
        # spill 文件存在
        for path in store._spilled.values():
            assert os.path.exists(path)

    def test_getitem_from_memory(self, store):
        """从内存访问帧"""
        store.append(self.make_frame(100))
        store.append(self.make_frame(200))
        frame = store[1]
        assert frame[0, 0, 0] == 200

    def test_getitem_from_spill(self, store):
        """从磁盘懒加载帧"""
        for i in range(10):
            store.append(self.make_frame(i))
        # 第 0 帧应该已被 spill
        assert 0 in store._spilled
        frame = store[0]
        assert frame[0, 0, 0] == 0

    def test_cleanup_removes_spill_files(self, store, spill_dir):
        for i in range(10):
            store.append(self.make_frame(i))
        spill_files = list(Path(spill_dir).glob("*.jpg"))
        assert len(spill_files) > 0
        store.cleanup()
        spill_files_after = list(Path(spill_dir).glob("*.jpg"))
        assert len(spill_files_after) == 0
        # FrameStore 本身变为空
        assert len(store._spilled) == 0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_frame_io.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 FrameStore 和 encode_frame**

```python
# src/illegal_review/preprocessing_layer/utils/frame_io.py
from pathlib import Path
import cv2
import numpy as np
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.preprocessing_layer.exceptions import PreprocessingError


def encode_frame(data: np.ndarray, config: PreprocessingConfig) -> bytes:
    """将 np.ndarray 编码为 JPEG bytes"""
    # 灰度图转 RGB
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    success, buf = cv2.imencode(
        '.jpg', data,
        [cv2.IMWRITE_JPEG_QUALITY, config.frame_encode_quality]
    )
    if not success:
        raise PreprocessingError("Frame encoding failed")
    return buf.tobytes()


class FrameStore:
    """帧存储混合策略：低于阈值全内存，超出 FIFO spill JPEG 磁盘"""

    def __init__(self, memory_limit: int = 9000, spill_dir: str = ""):
        self._memory: list[np.ndarray] = []
        self._spilled: dict[int, Path] = {}  # frame_index → spill 文件路径
        self._spill_index_offset: int = 0    # _memory[0] 对应的全局帧索引
        self._limit = memory_limit
        self._spill_dir = Path(spill_dir)
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._count: int = 0
        self.spill_count: int = 0

    def append(self, frame: np.ndarray) -> None:
        """追加帧，超出阈值自动 FIFO spill"""
        self._memory.append(frame)
        self._count += 1
        if len(self._memory) > self._limit:
            self._spill_one()

    def _spill_one(self) -> None:
        """将最早的内存帧溢出到磁盘"""
        frame = self._memory.pop(0)
        global_idx = self._spill_index_offset
        self._spill_index_offset += 1

        spill_path = self._spill_dir / f"frame_{global_idx:08d}.jpg"
        cv2.imwrite(
            str(spill_path), frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        self._spilled[global_idx] = spill_path
        self.spill_count += 1

    def __getitem__(self, idx: int) -> np.ndarray:
        """懒加载：磁盘帧读回，内存帧直接返回"""
        if idx < 0:
            idx += self._count
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Frame index {idx} out of range [0, {self._count})")
        if idx in self._spilled:
            frame = cv2.imread(str(self._spilled[idx]))
            if frame is None:
                raise PreprocessingError(f"Failed to read spilled frame {idx}")
            return frame
        mem_idx = idx - self._spill_index_offset
        return self._memory[mem_idx]

    def __len__(self) -> int:
        return self._count

    def cleanup(self) -> None:
        """删除所有 spill 文件并清空状态"""
        for path in self._spilled.values():
            path.unlink(missing_ok=True)
        self._spilled.clear()
        self._memory.clear()
        self._count = 0
        self._spill_index_offset = 0
        self.spill_count = 0
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_frame_io.py -v
```

Expected: PASS (8/8)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/utils/__init__.py src/illegal_review/preprocessing_layer/utils/frame_io.py tests/preprocessing_layer/test_frame_io.py
git commit -m "feat: add FrameStore with FIFO spill strategy and encode_frame utility"
```

---

### Task 6: FFmpegHelper

**Files:**
- Create: `src/illegal_review/preprocessing_layer/utils/ffmpeg_helper.py`
- Create: `tests/preprocessing_layer/test_ffmpeg_helper.py`

- [ ] **Step 1: 编写 FFmpegHelper 测试（mock subprocess）**

```python
# tests/preprocessing_layer/test_ffmpeg_helper.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.preprocessing_layer.utils.ffmpeg_helper import FFmpegHelper
from src.illegal_review.data_models import VideoMetadata


class TestFFmpegHelper:
    @pytest.fixture
    def helper(self):
        return FFmpegHelper(ffmpeg_path="ffmpeg", ffprobe_path="ffprobe")

    @pytest.mark.asyncio
    async def test_probe_video_returns_metadata(self, helper):
        """ffprobe 成功返回 VideoMetadata"""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(
            b'{"streams":[{"codec_type":"video","duration":"60.0",'
            b'"r_frame_rate":"30/1","width":1920,"height":1080,'
            b'"codec_name":"h264","bit_rate":"2000000"},'
            b'{"codec_type":"audio","codec_name":"aac"}]}',
            b""
        ))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.probe_video("/tmp/test.mp4")
            assert result.duration == 60.0
            assert result.fps == 30.0
            assert result.width == 1920
            assert result.height == 1080
            assert result.codec == "h264"
            assert result.audio_codec == "aac"

    @pytest.mark.asyncio
    async def test_probe_video_timeout(self, helper):
        """ffprobe 超时抛异常"""
        with patch("asyncio.wait_for", AsyncMock(side_effect=TimeoutError())):
            with pytest.raises(Exception):
                await helper.probe_video("/tmp/test.mp4")

    @pytest.mark.asyncio
    async def test_probe_video_nonzero_return(self, helper):
        """ffprobe 返回非零退出码"""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"file not found"))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.probe_video("/tmp/missing.mp4")
            assert result is None

    @pytest.mark.asyncio
    async def test_extract_audio_success(self, helper):
        """音频提取成功"""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.extract_audio("/tmp/test.mp4", "/tmp/out.wav", 16000)
            assert result == "/tmp/out.wav"

    @pytest.mark.asyncio
    async def test_extract_audio_no_stream_graceful(self, helper):
        """无音轨时静默降级，不抛异常"""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1  # ffmpeg 对无音轨返回非零
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Stream map 'a' matches no streams"))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            result = await helper.extract_audio("/tmp/no_audio.mp4", "/tmp/out.wav", 16000)
            assert result is None  # 降级返回 None

    @pytest.mark.asyncio
    async def test_decode_frames_pipe_mode(self, helper):
        """decode_frames 通过 pipe 模式解码"""
        import numpy as np
        from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore

        # 构造一个极小的 rawvideo 帧 (64x48 RGB)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        raw_data = frame.tobytes()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.read = AsyncMock(side_effect=[raw_data, b""])  # 读一帧后EOF

        store = FrameStore(memory_limit=100, spill_dir="/tmp/test_spill")

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            await helper.decode_frames(
                "/tmp/test.mp4", store,
                width=64, height=48, fps=30.0
            )
            # 解码了一帧
            assert len(store) == 1

    @pytest.mark.asyncio
    async def test_decode_frames_empty_video(self, helper):
        """空视频/损坏视频"""
        from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore
        from src.illegal_review.preprocessing_layer.exceptions import DecodeError

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Invalid data found"))

        store = FrameStore(memory_limit=100, spill_dir="/tmp/test_spill")

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            with pytest.raises(DecodeError):
                await helper.decode_frames(
                    "/tmp/bad.mp4", store,
                    width=1920, height=1080, fps=30.0
                )
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_ffmpeg_helper.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 FFmpegHelper**

```python
# src/illegal_review/preprocessing_layer/utils/ffmpeg_helper.py
import asyncio
import json
import logging
import os
from typing import Optional
import numpy as np

from src.illegal_review.data_models import VideoMetadata
from src.illegal_review.preprocessing_layer.exceptions import DecodeError
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore

logger = logging.getLogger(__name__)


class FFmpegHelper:
    """FFmpeg 命令封装 — 子进程安全模式"""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    async def probe_video(self, path: str, timeout: int = 20) -> Optional[VideoMetadata]:
        """ffprobe 获取视频元数据（兜底用，优先使用 InputResult 透传的 metadata）"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffprobe, "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error(f"ffprobe timeout for {path}")
                return None
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()

            if proc.returncode != 0:
                logger.warning(f"ffprobe failed for {path}: {stderr[-500:].decode(errors='replace')}")
                return None

            data = json.loads(stdout.decode())
            video_stream = None
            audio_stream = None
            for s in data.get("streams", []):
                if s.get("codec_type") == "video" and video_stream is None:
                    video_stream = s
                elif s.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = s

            if video_stream is None:
                return None

            fps_str = video_stream.get("r_frame_rate", "0/1")
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0

            return VideoMetadata(
                duration=float(data.get("format", {}).get("duration", video_stream.get("duration", 0))),
                fps=fps,
                width=video_stream.get("width", 0),
                height=video_stream.get("height", 0),
                codec=video_stream.get("codec_name", "unknown"),
                audio_codec=audio_stream.get("codec_name") if audio_stream else None,
                bitrate=int(data.get("format", {}).get("bit_rate", 0)) or None,
            )
        except Exception as e:
            logger.error(f"ffprobe error for {path}: {e}")
            return None

    async def extract_audio(
        self, video_path: str, output_path: str, sample_rate: int = 16000, timeout: int = 60
    ) -> Optional[str]:
        """提取音频轨道 → 16kHz 单声道 PCM WAV，无音轨时返回 None"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffmpeg, "-y",
                "-i", video_path,
                "-vn",                          # 禁用视频
                "-acodec", "pcm_s16le",         # PCM 16-bit
                "-ar", str(sample_rate),         # 重采样
                "-ac", "1",                      # 单声道
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error(f"Audio extraction timeout for {video_path}")
                return None
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()

            if proc.returncode != 0:
                err_msg = stderr[-500:].decode(errors="replace")
                # "Stream map 'a' matches no streams" → 无音轨，正常降级
                if "matches no streams" in err_msg.lower() or "audio" in err_msg.lower():
                    logger.info(f"No audio stream in {video_path}")
                    return None
                logger.warning(f"Audio extraction warning for {video_path}: {err_msg}")
                return None

            return output_path
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            return None

    async def decode_frames(
        self, video_path: str, store: FrameStore,
        width: int, height: int, fps: float, timeout: Optional[int] = None
    ) -> None:
        """FFmpeg pipe 模式解码视频帧 → FrameStore"""
        frame_size = width * height * 3  # rgb24
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffmpeg, "-y",
                "-i", video_path,
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                # 逐帧读取 pipe（非阻塞读，每帧固定字节）
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            proc.stdout.read(frame_size), timeout=timeout or 300
                        )
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        raise DecodeError(f"Decode timeout after {len(store)} frames")

                    if not raw or len(raw) < frame_size:
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                    store.append(frame)

                await proc.wait()
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()

            if len(store) == 0:
                stderr = await proc.stderr.read() if proc.stderr else b""
                raise DecodeError(
                    f"No frames decoded from {video_path}: "
                    f"{stderr[-500:].decode(errors='replace')}"
                )

        except DecodeError:
            raise
        except Exception as e:
            raise DecodeError(f"Decode failed for {video_path}: {e}")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_ffmpeg_helper.py -v
```

Expected: PASS (7/7)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/utils/ffmpeg_helper.py tests/preprocessing_layer/test_ffmpeg_helper.py
git commit -m "feat: add FFmpegHelper with pipe-mode decode, audio extract, and ffprobe"
```

---

### Task 7: TempCleaner

**Files:**
- Create: `src/illegal_review/preprocessing_layer/utils/temp_cleaner.py`
- Create: `tests/preprocessing_layer/test_temp_cleaner.py`

- [ ] **Step 1: 编写 TempCleaner 测试**

```python
# tests/preprocessing_layer/test_temp_cleaner.py
import pytest
import os
import tempfile
from pathlib import Path
from uuid import uuid4
from src.illegal_review.preprocessing_layer.utils.temp_cleaner import TempCleaner
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class TestTempCleaner:
    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def ctx(self, temp_dir):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_files(self, ctx, temp_dir):
        """清理后临时目录为空"""
        # 创建模拟 spill 目录
        spill_dir = Path(temp_dir) / "spill"
        spill_dir.mkdir()
        (spill_dir / "frame_00000000.jpg").write_bytes(b"fake jpeg")
        assert spill_dir.exists()

        # 创建模拟音频文件
        audio_path = Path(temp_dir) / "audio.wav"
        audio_path.write_bytes(b"fake wav")
        ctx.audio_path = str(audio_path)

        cleaner = TempCleaner(ctx, temp_dir=str(temp_dir))
        async with cleaner:
            pass  # 正常退出

        # 清理后文件不存在
        assert not audio_path.exists()
        assert not spill_dir.exists() or not list(spill_dir.glob("*.jpg"))

    @pytest.mark.asyncio
    async def test_cleanup_not_called_when_disabled(self, ctx, temp_dir):
        """temp_cleanup_enabled=False 时跳过清理"""
        ctx.config.temp_cleanup_enabled = False
        audio_path = Path(temp_dir) / "audio.wav"
        audio_path.write_bytes(b"fake wav")
        ctx.audio_path = str(audio_path)

        cleaner = TempCleaner(ctx, temp_dir=str(temp_dir))
        async with cleaner:
            pass

        # 文件仍然存在
        assert audio_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_swallows_errors(self, ctx, temp_dir):
        """清理异常不传播（已被删除/权限等）"""
        ctx.audio_path = "/nonexistent/path/audio.wav"
        cleaner = TempCleaner(ctx, temp_dir=str(temp_dir))
        # 不应抛异常
        async with cleaner:
            pass
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_temp_cleaner.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 TempCleaner**

```python
# src/illegal_review/preprocessing_layer/utils/temp_cleaner.py
import logging
from pathlib import Path
from src.illegal_review.preprocessing_layer.context import PipelineContext

logger = logging.getLogger(__name__)


class TempCleaner:
    """异步上下文管理器 — 绑定 Pipeline 生命周期，自动清理临时文件"""

    def __init__(self, ctx: PipelineContext, temp_dir: str = ""):
        self._ctx = ctx
        self._temp_dir = Path(temp_dir) if temp_dir else None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._ctx.config.temp_cleanup_enabled:
            return
        self._cleanup()
        return False  # 不吞异常

    def _cleanup(self) -> None:
        """清理所有中间产物（吞掉所有异常）"""
        try:
            # 清理 spill 文件
            if self._ctx.raw_frames is not None:
                self._ctx.raw_frames.cleanup()

            # 清理音频临时文件
            if self._ctx.audio_path:
                try:
                    Path(self._ctx.audio_path).unlink(missing_ok=True)
                except OSError:
                    pass

            # 清理 spill 目录（如果为空）
            if self._temp_dir:
                spill_dir = self._temp_dir / "spill"
                if spill_dir.exists():
                    try:
                        spill_dir.rmdir()  # 只在目录为空时删除
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"TempCleaner: cleanup error (non-fatal): {e}")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_temp_cleaner.py -v
```

Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/utils/temp_cleaner.py tests/preprocessing_layer/test_temp_cleaner.py
git commit -m "feat: add TempCleaner async context manager"
```

---

### Task 8: DecodeStage

**Files:**
- Create: `src/illegal_review/preprocessing_layer/stages/__init__.py` (空文件)
- Create: `src/illegal_review/preprocessing_layer/stages/decode_stage.py`
- Create: `tests/preprocessing_layer/test_decode_stage.py`

- [ ] **Step 1: 编写 DecodeStage 测试**

```python
# tests/preprocessing_layer/test_decode_stage.py
import pytest
import tempfile
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.preprocessing_layer.stages.decode_stage import DecodeStage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.data_models import VideoMetadata


class TestDecodeStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(frame_store_memory_limit=100)

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=config,
            _input_metadata=VideoMetadata(
                duration=10.0, fps=30.0, width=1920, height=1080, codec="h264"
            ),
        )

    def test_name_and_dependencies(self, config):
        stage = DecodeStage(config)
        assert stage.name == "decode"
        assert stage.dependencies == []

    @pytest.mark.asyncio
    async def test_decode_uses_input_metadata(self, config, ctx):
        """使用 _input_metadata 中的参数调用 FFmpeg"""
        stage = DecodeStage(config)
        mock_helper = AsyncMock()
        mock_helper.decode_frames = AsyncMock()

        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)

        # 校验传参使用 _input_metadata
        call_kwargs = mock_helper.decode_frames.call_args
        assert call_kwargs[0][0] == "/tmp/test.mp4"  # video_path
        assert call_kwargs[1]["width"] == 1920
        assert call_kwargs[1]["height"] == 1080
        assert call_kwargs[1]["fps"] == 30.0
        # FrameStore 创建了
        assert ctx.raw_frames is not None
        assert ctx.fps == 30.0

    @pytest.mark.asyncio
    async def test_decode_falls_back_to_probe(self, config, ctx):
        """_input_metadata 缺失时降级到独立 ffprobe"""
        ctx._input_metadata = None
        stage = DecodeStage(config)
        mock_helper = AsyncMock()
        mock_helper.probe_video = AsyncMock(return_value=VideoMetadata(
            duration=5.0, fps=25.0, width=640, height=480, codec="h264"
        ))
        mock_helper.decode_frames = AsyncMock()

        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)

        mock_helper.probe_video.assert_called_once()
        mock_helper.decode_frames.assert_called_once()

    @pytest.mark.asyncio
    async def test_decode_no_metadata_and_probe_fails(self, config, ctx):
        """metadata 缺失且 probe 也失败 → 抛 DecodeError"""
        from src.illegal_review.preprocessing_layer.exceptions import DecodeError

        ctx._input_metadata = None
        stage = DecodeStage(config)
        mock_helper = AsyncMock()
        mock_helper.probe_video = AsyncMock(return_value=None)

        with patch.object(stage, "_ffmpeg", mock_helper):
            with pytest.raises(DecodeError):
                await stage.process(ctx)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_decode_stage.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 DecodeStage**

```python
# src/illegal_review/preprocessing_layer/stages/decode_stage.py
import logging
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.preprocessing_layer.utils.ffmpeg_helper import FFmpegHelper
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore
from src.illegal_review.preprocessing_layer.exceptions import DecodeError

logger = logging.getLogger(__name__)


class DecodeStage(Stage):
    """视频解码阶段 — FFmpeg pipe 模式解码 → FrameStore"""

    @property
    def name(self) -> str:
        return "decode"

    @property
    def dependencies(self) -> list[str]:
        return []

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        self._ffmpeg = FFmpegHelper(
            ffmpeg_path=config.ffmpeg_path or "ffmpeg",
            ffprobe_path=config.ffmpeg_path or "ffprobe",
        )

    async def process(self, ctx: PipelineContext) -> None:
        # 获取元数据（优先透传，缺失时独立 probe）
        metadata = ctx._input_metadata
        if metadata is None:
            metadata = await self._ffmpeg.probe_video(ctx.video_path)
        if metadata is None:
            raise DecodeError(f"Cannot obtain video metadata for {ctx.video_path}")

        # 创建 FrameStore
        store = FrameStore(
            memory_limit=self._config.frame_store_memory_limit,
            spill_dir=f"{ctx.video_path}_spill",
        )

        await self._ffmpeg.decode_frames(
            ctx.video_path, store,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
        )

        ctx.raw_frames = store
        ctx.fps = metadata.fps
        ctx.total_frames = len(store)
        ctx.stats["total_frames"] = float(ctx.total_frames)
        ctx.stats["spill_count"] = float(store.spill_count)
        logger.info(f"Decoded {ctx.total_frames} frames from {ctx.video_path}")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_decode_stage.py -v
```

Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/stages/__init__.py src/illegal_review/preprocessing_layer/stages/decode_stage.py tests/preprocessing_layer/test_decode_stage.py
git commit -m "feat: add DecodeStage with pipe-mode FFmpeg decoding"
```

---

### Task 9: AudioExtractStage

**Files:**
- Create: `src/illegal_review/preprocessing_layer/stages/audio_extract_stage.py`
- Create: `tests/preprocessing_layer/test_audio_extract_stage.py`

- [ ] **Step 1: 编写 AudioExtractStage 测试**

```python
# tests/preprocessing_layer/test_audio_extract_stage.py
import pytest
import tempfile
from uuid import uuid4
from unittest.mock import AsyncMock, patch
from src.illegal_review.preprocessing_layer.stages.audio_extract_stage import AudioExtractStage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class TestAudioExtractStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(audio_sample_rate=16000)

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=config,
        )

    def test_name_and_dependencies(self, config):
        stage = AudioExtractStage(config)
        assert stage.name == "audio_extract"
        assert stage.dependencies == ["decode"]

    @pytest.mark.asyncio
    async def test_extract_audio_success(self, config, ctx):
        """有音轨：ctx.audio_path 被设置"""
        stage = AudioExtractStage(config)
        mock_helper = AsyncMock()
        mock_helper.extract_audio = AsyncMock(return_value="/tmp/audio.wav")

        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)

        assert ctx.audio_path == "/tmp/audio.wav"

    @pytest.mark.asyncio
    async def test_extract_audio_no_track(self, config, ctx):
        """无音轨：静默降级，audio_path 保持 None"""
        stage = AudioExtractStage(config)
        mock_helper = AsyncMock()
        mock_helper.extract_audio = AsyncMock(return_value=None)

        with patch.object(stage, "_ffmpeg", mock_helper):
            await stage.process(ctx)

        assert ctx.audio_path is None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_audio_extract_stage.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 AudioExtractStage**

```python
# src/illegal_review/preprocessing_layer/stages/audio_extract_stage.py
import logging
import tempfile
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig
from src.illegal_review.preprocessing_layer.utils.ffmpeg_helper import FFmpegHelper

logger = logging.getLogger(__name__)


class AudioExtractStage(Stage):
    """音频提取阶段 — FFmpeg 提取 → 16kHz 单声道 PCM WAV"""

    @property
    def name(self) -> str:
        return "audio_extract"

    @property
    def dependencies(self) -> list[str]:
        return ["decode"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        self._ffmpeg = FFmpegHelper(ffmpeg_path=config.ffmpeg_path or "ffmpeg")

    async def process(self, ctx: PipelineContext) -> None:
        output_path = f"{ctx.video_path}_audio.wav"
        result = await self._ffmpeg.extract_audio(
            ctx.video_path,
            output_path,
            sample_rate=self._config.audio_sample_rate,
        )
        if result is not None:
            ctx.audio_path = result
            logger.info(f"Audio extracted to {result}")
        else:
            logger.info(f"No audio track in {ctx.video_path} — silent degradation")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_audio_extract_stage.py -v
```

Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/stages/audio_extract_stage.py tests/preprocessing_layer/test_audio_extract_stage.py
git commit -m "feat: add AudioExtractStage with silent degradation for no-audio videos"
```

---

### Task 10: FrameSampleStage

**Files:**
- Create: `src/illegal_review/preprocessing_layer/stages/frame_sample_stage.py`
- Create: `tests/preprocessing_layer/test_frame_sample_stage.py`

- [ ] **Step 1: 编写 FrameSampleStage 测试**

```python
# tests/preprocessing_layer/test_frame_sample_stage.py
import pytest
import numpy as np
from uuid import uuid4
from src.illegal_review.preprocessing_layer.stages.frame_sample_stage import FrameSampleStage
from src.illegal_review.preprocessing_layer.context import PipelineContext, SampledFrame
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore
from src.illegal_review.config.settings import PreprocessingConfig


class TestFrameSampleStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(
            frame_sample_interval_short=1,
            frame_sample_interval_long=2,
            short_video_threshold=60,
            scene_change_threshold=30,
        )

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=config,
        )

    def test_name_and_dependencies(self, config):
        stage = FrameSampleStage(config)
        assert stage.name == "frame_sample"
        assert stage.dependencies == ["decode"]

    def test_short_video_interval(self, config, ctx):
        """短视频 (<60s)：每秒 1 帧"""
        ctx.fps = 30.0
        ctx.total_frames = 900  # 30s video
        store = FrameStore(memory_limit=1000, spill_dir="/tmp/spill")
        for i in range(900):
            store.append(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        ctx.raw_frames = store

        stage = FrameSampleStage(config)
        import asyncio
        asyncio.run(stage.process(ctx))

        # 30帧 @1fps = 30 sampled frames
        assert ctx.sampled_frames is not None
        # 实际采样数接近 30 (± 场景检测增加额外帧)
        assert 25 <= len(ctx.sampled_frames) <= 50

    def test_long_video_interval(self, config, ctx):
        """长视频 (≥60s)：每 2 秒 1 帧"""
        ctx.fps = 30.0
        ctx.total_frames = 3600  # 120s video
        store = FrameStore(memory_limit=5000, spill_dir="/tmp/spill")
        for i in range(3600):
            store.append(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        ctx.raw_frames = store

        stage = FrameSampleStage(config)
        import asyncio
        asyncio.run(stage.process(ctx))

        assert ctx.sampled_frames is not None
        # 60帧 @0.5fps = ~60 sampled frames
        assert 50 <= len(ctx.sampled_frames) <= 100

    def test_scene_change_detection(self, config, ctx):
        """场景切换帧被额外保留"""
        ctx.fps = 10.0
        ctx.total_frames = 30
        store = FrameStore(memory_limit=100, spill_dir="/tmp/spill")

        # 构造 30 帧：每 10 帧画面突变
        for i in range(30):
            if i < 10:
                frame = np.full((64, 64, 3), 10, dtype=np.uint8)
            elif i < 20:
                frame = np.full((64, 64, 3), 200, dtype=np.uint8)
            else:
                frame = np.full((64, 64, 3), 100, dtype=np.uint8)
            store.append(frame)
        ctx.raw_frames = store

        stage = FrameSampleStage(config)
        import asyncio
        asyncio.run(stage.process(ctx))

        assert ctx.sampled_frames is not None
        # 应该有常规采样帧 + 场景边界帧
        assert len(ctx.sampled_frames) > 0

    def test_empty_video_graceful(self, config, ctx):
        """空帧序列 → sampled_frames 保持 None"""
        ctx.fps = 30.0
        ctx.total_frames = 0
        store = FrameStore(memory_limit=100, spill_dir="/tmp/spill")
        ctx.raw_frames = store

        stage = FrameSampleStage(config)
        import asyncio
        asyncio.run(stage.process(ctx))

        assert ctx.sampled_frames is None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_frame_sample_stage.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 FrameSampleStage**

```python
# src/illegal_review/preprocessing_layer/stages/frame_sample_stage.py
import logging
import numpy as np
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext, SampledFrame
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class FrameSampleStage(Stage):
    """帧采样 + 场景变化检测"""

    @property
    def name(self) -> str:
        return "frame_sample"

    @property
    def dependencies(self) -> list[str]:
        return ["decode"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config

    async def process(self, ctx: PipelineContext) -> None:
        if ctx.raw_frames is None or len(ctx.raw_frames) == 0:
            logger.warning("No frames to sample")
            return

        fps = ctx.fps or 30.0
        total_frames = len(ctx.raw_frames)
        duration = total_frames / fps if fps > 0 else 0

        # 选择采样间隔
        if duration < self._config.short_video_threshold:
            interval_seconds = self._config.frame_sample_interval_short
        else:
            interval_seconds = self._config.frame_sample_interval_long

        interval_frames = max(1, int(fps * interval_seconds))
        threshold = self._config.scene_change_threshold

        sampled: list[SampledFrame] = []
        prev_frame = None

        for i in range(total_frames):
            frame = ctx.raw_frames[i]
            timestamp = i / fps if fps > 0 else float(i)

            is_sample = (i % interval_frames == 0)

            # 场景变化检测
            if not is_sample and prev_frame is not None:
                diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32)).mean()
                if diff > threshold:
                    is_sample = True

            if is_sample:
                sampled.append(SampledFrame(
                    frame_index=i,
                    timestamp=timestamp,
                    data=frame.copy(),
                ))

            prev_frame = frame

        ctx.sampled_frames = sampled if sampled else None
        ctx.stats["sampled_frames"] = float(len(sampled))
        logger.info(f"Sampled {len(sampled)} frames from {total_frames} total")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_frame_sample_stage.py -v
```

Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/stages/frame_sample_stage.py tests/preprocessing_layer/test_frame_sample_stage.py
git commit -m "feat: add FrameSampleStage with adaptive interval and scene change detection"
```

---

### Task 11: SpeechStage

**Files:**
- Create: `src/illegal_review/preprocessing_layer/stages/speech_stage.py`
- Create: `tests/preprocessing_layer/test_speech_stage.py`

- [ ] **Step 1: 编写 SpeechStage 测试**

```python
# tests/preprocessing_layer/test_speech_stage.py
import pytest
from uuid import uuid4
from unittest.mock import patch, MagicMock
from src.illegal_review.preprocessing_layer.stages.speech_stage import SpeechStage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class TestSpeechStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(whisper_model="small")

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=config,
        )

    def test_name_and_dependencies(self, config):
        with patch("whisper.load_model", MagicMock()):
            stage = SpeechStage(config)
            assert stage.name == "speech"
            assert stage.dependencies == ["audio_extract"]

    def test_preloads_whisper_on_init(self, config):
        """构造时预加载 Whisper 模型"""
        mock_load = MagicMock()
        with patch("whisper.load_model", mock_load):
            SpeechStage(config)
            mock_load.assert_called_once_with("small")

    @pytest.mark.asyncio
    async def test_transcribe_success(self, config, ctx):
        """有音频路径时正常转写"""
        ctx.audio_path = "/tmp/audio.wav"

        mock_model = MagicMock()
        mock_result = {"text": "测试文本内容", "segments": [{"text": "测试", "start": 0.0, "end": 1.0}]}
        mock_model.transcribe = MagicMock(return_value=mock_result)

        with patch("whisper.load_model", MagicMock(return_value=mock_model)):
            stage = SpeechStage(config)
            stage._model = mock_model

            await stage.process(ctx)

        assert ctx.transcript == "测试文本内容"
        assert ctx.transcript_segments is not None
        assert len(ctx.transcript_segments) == 1

    @pytest.mark.asyncio
    async def test_no_audio_skips(self, config, ctx):
        """无音频路径 → 跳过，transcript 保持 None"""
        ctx.audio_path = None

        with patch("whisper.load_model", MagicMock()):
            stage = SpeechStage(config)
            await stage.process(ctx)

        assert ctx.transcript is None
        assert ctx.transcript_segments is None

    @pytest.mark.asyncio
    async def test_transcribe_failure_graceful(self, config, ctx):
        """转写失败 → transcript 保持 None，记录 error"""
        ctx.audio_path = "/tmp/audio.wav"

        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        with patch("whisper.load_model", MagicMock(return_value=mock_model)):
            stage = SpeechStage(config)
            stage._model = mock_model

            await stage.process(ctx)

        assert ctx.transcript is None
        assert len(ctx.errors) > 0
        assert "CUDA OOM" in ctx.errors[0]
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_speech_stage.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 SpeechStage**

```python
# src/illegal_review/preprocessing_layer/stages/speech_stage.py
import asyncio
import logging
import whisper
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class SpeechStage(Stage):
    """Whisper 语音转写阶段"""

    @property
    def name(self) -> str:
        return "speech"

    @property
    def dependencies(self) -> list[str]:
        return ["audio_extract"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        # 构造时预加载模型
        logger.info(f"Loading Whisper model: {config.whisper_model}")
        self._model = whisper.load_model(config.whisper_model)
        logger.info(f"Whisper model '{config.whisper_model}' loaded")

    async def process(self, ctx: PipelineContext) -> None:
        if ctx.audio_path is None:
            logger.info("No audio path — skipping speech recognition")
            return

        try:
            # CPU 密集型推理 → 线程池
            result = await asyncio.to_thread(
                self._model.transcribe, ctx.audio_path
            )
            ctx.transcript = result.get("text", "")
            ctx.transcript_segments = result.get("segments")
            logger.info(f"Transcription complete: {len(ctx.transcript)} chars")
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            ctx.errors.append(f"SpeechStage error: {e}")
            # transcript/segments 保持 None
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_speech_stage.py -v
```

Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/stages/speech_stage.py tests/preprocessing_layer/test_speech_stage.py
git commit -m "feat: add SpeechStage with Whisper small preloading and asyncio.to_thread"
```

---

### Task 12: OCRStage

**Files:**
- Create: `src/illegal_review/preprocessing_layer/stages/ocr_stage.py`
- Create: `tests/preprocessing_layer/test_ocr_stage.py`

- [ ] **Step 1: 编写 OCRStage 测试**

```python
# tests/preprocessing_layer/test_ocr_stage.py
import pytest
import numpy as np
from uuid import uuid4
from unittest.mock import patch, MagicMock
from src.illegal_review.preprocessing_layer.stages.ocr_stage import OCRStage
from src.illegal_review.preprocessing_layer.context import PipelineContext, SampledFrame
from src.illegal_review.config.settings import PreprocessingConfig


class TestOCRStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(ocr_languages=["ch_sim", "en"])

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=config,
        )

    def test_name_and_dependencies(self, config):
        with patch("easyocr.Reader", MagicMock()):
            stage = OCRStage(config)
            assert stage.name == "ocr"
            assert stage.dependencies == ["frame_sample"]

    def test_preloads_easyocr_on_init(self, config):
        """构造时预加载 EasyOCR"""
        mock_reader = MagicMock()
        with patch("easyocr.Reader", mock_reader):
            OCRStage(config)
            mock_reader.assert_called_once_with(["ch_sim", "en"])

    @pytest.mark.asyncio
    async def test_ocr_single_frame(self, config, ctx):
        """对单个采样帧执行 OCR"""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        ctx.sampled_frames = [
            SampledFrame(frame_index=0, timestamp=0.0, data=frame)
        ]

        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(return_value=[
            ([[0, 0], [100, 0], [100, 30], [0, 30]], "检测文字", 0.95)
        ])

        with patch("easyocr.Reader", MagicMock(return_value=mock_reader)):
            stage = OCRStage(config)
            stage._reader = mock_reader

            await stage.process(ctx)

        assert ctx.ocr_results is not None
        assert len(ctx.ocr_results) == 1
        assert ctx.ocr_results[0].text == "检测文字"
        assert ctx.ocr_results[0].confidence == 0.95
        assert ctx.ocr_results[0].frame_index == 0

    @pytest.mark.asyncio
    async def test_no_sampled_frames_skips(self, config, ctx):
        """无采样帧 → 跳过"""
        ctx.sampled_frames = None

        with patch("easyocr.Reader", MagicMock()):
            stage = OCRStage(config)
            await stage.process(ctx)

        assert ctx.ocr_results is None

    @pytest.mark.asyncio
    async def test_ocr_failure_graceful(self, config, ctx):
        """OCR 失败 → 记录 error"""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        ctx.sampled_frames = [
            SampledFrame(frame_index=0, timestamp=0.0, data=frame)
        ]

        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(side_effect=RuntimeError("OCR engine error"))

        with patch("easyocr.Reader", MagicMock(return_value=mock_reader)):
            stage = OCRStage(config)
            stage._reader = mock_reader

            await stage.process(ctx)

        assert ctx.ocr_results is None
        assert len(ctx.errors) > 0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_ocr_stage.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 OCRStage**

```python
# src/illegal_review/preprocessing_layer/stages/ocr_stage.py
import asyncio
import logging
import easyocr
from src.illegal_review.data_models import OCRResult as OCRResultModel
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class OCRStage(Stage):
    """EasyOCR 文字识别阶段"""

    @property
    def name(self) -> str:
        return "ocr"

    @property
    def dependencies(self) -> list[str]:
        return ["frame_sample"]

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        # 构造时预加载模型
        logger.info(f"Loading EasyOCR: languages={config.ocr_languages}")
        self._reader = easyocr.Reader(config.ocr_languages)
        logger.info("EasyOCR loaded")

    async def process(self, ctx: PipelineContext) -> None:
        if ctx.sampled_frames is None:
            logger.info("No sampled frames — skipping OCR")
            return

        results: list[OCRResultModel] = []
        try:
            for frame in ctx.sampled_frames:
                # CPU 密集型推理 → 线程池
                detections = await asyncio.to_thread(
                    self._reader.readtext, frame.data
                )
                for bbox, text, confidence in detections:
                    # 扁平化 bbox
                    flat_bbox = [int(v) for pt in bbox for v in pt] if bbox else None
                    results.append(OCRResultModel(
                        text=text,
                        confidence=float(confidence),
                        bbox=flat_bbox,
                        frame_index=frame.frame_index,
                    ))

            ctx.ocr_results = results if results else None
            logger.info(f"OCR complete: {len(results)} text regions found")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            ctx.errors.append(f"OCRStage error: {e}")
            # ocr_results 保持 None
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_ocr_stage.py -v
```

Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/stages/ocr_stage.py tests/preprocessing_layer/test_ocr_stage.py
git commit -m "feat: add OCRStage with EasyOCR preloading and asyncio.to_thread"
```

---

### Task 13: Pipeline 调度器

**Files:**
- Create: `src/illegal_review/preprocessing_layer/pipeline.py`
- Create: `tests/preprocessing_layer/test_pipeline.py`

- [ ] **Step 1: 编写 Pipeline 测试**

```python
# tests/preprocessing_layer/test_pipeline.py
import pytest
import asyncio
from uuid import uuid4
from src.illegal_review.preprocessing_layer.pipeline import Pipeline
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.preprocessing_layer.exceptions import PipelineError
from src.illegal_review.config.settings import PreprocessingConfig


class _PassStage(Stage):
    """总是成功的 Stage"""
    def __init__(self, name, deps=None):
        self._name = name
        self._deps = deps or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def process(self, ctx):
        ctx.stats[f"{self._name}_done"] = 1.0


class _FailStage(Stage):
    """总是失败的 Stage（失败传播不抛异常）"""
    def __init__(self, name, deps=None):
        self._name = name
        self._deps = deps or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def process(self, ctx):
        raise RuntimeError(f"{self._name} failed")


class _RecordStage(Stage):
    """记录执行顺序的 Stage"""
    _exec_order = []

    def __init__(self, name, deps=None):
        self._name = name
        self._deps = deps or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    async def process(self, ctx):
        _RecordStage._exec_order.append(self._name)
        await asyncio.sleep(0.01)  # 让出控制权


class TestPipeline:
    @pytest.fixture
    def ctx(self):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    def test_single_stage(self, ctx):
        """单 Stage Pipeline"""
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("only_stage"))
        asyncio.run(pipeline.run(ctx))
        assert ctx.stats["only_stage_done"] == 1.0

    def test_linear_chain(self, ctx):
        """线性链: a → b → c"""
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("a"))
        pipeline.add_stage(_PassStage("b", deps=["a"]))
        pipeline.add_stage(_PassStage("c", deps=["b"]))
        asyncio.run(pipeline.run(ctx))
        assert ctx.stats["a_done"] == 1.0
        assert ctx.stats["b_done"] == 1.0
        assert ctx.stats["c_done"] == 1.0

    def test_parallel_branches(self, ctx):
        """并行分支互相独立"""
        _RecordStage._exec_order.clear()
        pipeline = Pipeline()
        pipeline.add_stage(_RecordStage("root"))
        pipeline.add_stage(_RecordStage("left", deps=["root"]))
        pipeline.add_stage(_RecordStage("right", deps=["root"]))

        asyncio.run(pipeline.run(ctx))
        # root 先执行，left 和 right 在 root 之后执行（并行）
        exec_order = _RecordStage._exec_order
        assert exec_order[0] == "root"
        assert "left" in exec_order[1:] and "right" in exec_order[1:]

    def test_upstream_failure_skips_downstream(self, ctx):
        """上游失败 → 下游跳过"""
        pipeline = Pipeline()
        pipeline.add_stage(_FailStage("root"))
        pipeline.add_stage(_PassStage("child", deps=["root"]))

        asyncio.run(pipeline.run(ctx))
        # child 没有执行
        assert "child_done" not in ctx.stats
        # error 被记录
        assert len(ctx.errors) > 0

    def test_branch_isolation(self, ctx):
        """独立分支失败不影响其他分支"""
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("root"))
        pipeline.add_stage(_FailStage("bad_branch", deps=["root"]))
        pipeline.add_stage(_PassStage("good_branch", deps=["root"]))

        asyncio.run(pipeline.run(ctx))
        # good_branch 仍然执行成功
        assert ctx.stats["good_branch_done"] == 1.0
        assert "bad_branch_done" not in ctx.stats

    def test_circular_dependency_detected(self, ctx):
        """循环依赖检测"""
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("a", deps=["b"]))
        pipeline.add_stage(_PassStage("b", deps=["a"]))

        with pytest.raises(PipelineError, match="Circular"):
            asyncio.run(pipeline.run(ctx))

    def test_duplicate_name_raises(self):
        """重复名称抛异常"""
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("dup"))
        with pytest.raises(ValueError, match="already registered"):
            pipeline.add_stage(_PassStage("dup"))

    def test_stage_timing(self, ctx):
        """Pipeline 自动为每个 Stage 计时"""
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("timed"))
        asyncio.run(pipeline.run(ctx))
        assert "timed_duration_ms" in ctx.stats
        assert ctx.stats["timed_duration_ms"] >= 0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_pipeline.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 Pipeline**

```python
# src/illegal_review/preprocessing_layer/pipeline.py
import asyncio
import logging
import time
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.preprocessing_layer.exceptions import PipelineError, PreprocessingError

logger = logging.getLogger(__name__)


class Pipeline:
    """DAG Pipeline 调度器 — 拓扑并行执行 + 失败隔离"""

    def __init__(self):
        self._stages: dict[str, Stage] = {}

    def add_stage(self, stage: Stage) -> None:
        if stage.name in self._stages:
            raise ValueError(f"Stage '{stage.name}' already registered")
        self._stages[stage.name] = stage

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        completed: set[str] = set()
        failed: set[str] = set()
        all_names = set(self._stages.keys())

        # 验证所有依赖存在
        for stage in self._stages.values():
            for dep in stage.dependencies:
                if dep not in all_names:
                    raise PipelineError(
                        f"Stage '{stage.name}' depends on '{dep}' which is not registered"
                    )

        while len(completed) + len(failed) < len(self._stages):
            # 计算就绪集合：依赖全部满足 且 所有上游未失败
            ready = {
                name for name, stage in self._stages.items()
                if name not in completed
                and name not in failed
                and all(d in completed for d in stage.dependencies)
                and not any(d in failed for d in stage.dependencies)
            }

            if not ready:
                # 存在未完成节点但无一就绪 → 循环依赖或死锁
                remaining = all_names - completed - failed
                raise PipelineError(
                    f"Circular dependency or deadlock detected. "
                    f"Remaining stages: {remaining}"
                )

            ready_stages = [(name, self._stages[name]) for name in ready]

            # 并行执行当前轮
            results = await asyncio.gather(
                *[self._run_stage(name, stage, ctx) for name, stage in ready_stages],
                return_exceptions=True,
            )

            for (name, _), result in zip(ready_stages, results):
                if isinstance(result, Exception):
                    failed.add(name)
                    ctx.errors.append(f"{name}: {result}")
                    logger.error(f"Stage '{name}' failed: {result}")
                else:
                    completed.add(name)

        ctx.stats["total_duration_ms"] = (time.perf_counter() - t0) * 1000

        # decode 失败 → 整体失败
        if "decode" in failed:
            raise PreprocessingError(
                f"Decode stage failed, pipeline aborted. Errors: {ctx.errors}"
            )

        return ctx

    async def _run_stage(self, name: str, stage: Stage, ctx: PipelineContext) -> None:
        t0 = time.perf_counter()
        await stage.process(ctx)
        elapsed = (time.perf_counter() - t0) * 1000
        ctx.stats[f"{name}_duration_ms"] = elapsed
        logger.info(f"Stage '{name}' completed in {elapsed:.0f}ms")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_pipeline.py -v
```

Expected: PASS (8/8)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/preprocessing_layer/pipeline.py tests/preprocessing_layer/test_pipeline.py
git commit -m "feat: add Pipeline DAG scheduler with parallel execution and failure isolation"
```

---

### Task 14: PreprocessingService

**Files:**
- Create: `src/illegal_review/preprocessing_layer/service.py`
- Modify: `src/illegal_review/preprocessing_layer/__init__.py`
- Create: `tests/preprocessing_layer/test_service.py`

- [ ] **Step 1: 编写 Service 测试**

```python
# tests/preprocessing_layer/test_service.py
import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.preprocessing_layer.service import PreprocessingService
from src.illegal_review.preprocessing_layer.context import PipelineContext, SampledFrame
from src.illegal_review.data_models import InputResult, SourceInfo, VideoMetadata, PreprocessingResult
from src.illegal_review.config.settings import PreprocessingConfig


def make_input_result(video_path="/tmp/test.mp4", has_metadata=True):
    return InputResult(
        input_id=uuid4(),
        input_type="file",
        source_info=SourceInfo(
            original_source="test.mp4",
            file_size=1024,
            content_type="video/mp4",
        ),
        video_metadata=VideoMetadata(
            duration=10.0, fps=30.0, width=1920, height=1080, codec="h264"
        ) if has_metadata else None,
        temp_path=video_path,
        status="completed",
        created_at=datetime.now(timezone.utc),
        processed_at=datetime.now(timezone.utc),
    )


class TestPreprocessingService:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(temp_cleanup_enabled=False)

    @pytest.fixture
    def service(self, config):
        with patch("src.illegal_review.preprocessing_layer.stages.speech_stage.whisper.load_model", MagicMock()), \
             patch("src.illegal_review.preprocessing_layer.stages.ocr_stage.easyocr.Reader", MagicMock()):
            svc = PreprocessingService(config)
            return svc

    @pytest.mark.asyncio
    async def test_process_full_pipeline(self, service):
        """完整流水线处理"""
        input_result = make_input_result()

        # Mock 整个 Pipeline.run 避免实际调用 FFmpeg/Whisper/EasyOCR
        mock_ctx = PipelineContext(
            input_id=input_result.input_id,
            video_path=input_result.temp_path,
            config=service._config,
            _input_metadata=input_result.video_metadata,
            fps=30.0,
            total_frames=300,
            audio_path="/tmp/test_audio.wav",
            audio_duration=10.0,
            sampled_frames=[
                SampledFrame(
                    frame_index=0,
                    timestamp=0.0,
                    data=np.zeros((480, 640, 3), dtype=np.uint8),
                )
            ],
            transcript="测试转录文本",
            transcript_segments=[{"text": "测试", "start": 0.0, "end": 1.0}],
            ocr_results=[],
            stats={"decode_duration_ms": 100.0},
        )
        mock_ctx.raw_frames = MagicMock()
        mock_ctx.raw_frames.spill_count = 0

        with patch.object(service.pipeline, "run", AsyncMock(return_value=mock_ctx)):
            result = await service.process(input_result)

        assert isinstance(result, PreprocessingResult)
        assert result.input_id == input_result.input_id
        assert result.transcript == "测试转录文本"
        assert result.frames is not None
        assert len(result.frames) == 1
        assert isinstance(result.frames[0].image_data, bytes)
        assert result.audio is not None
        assert result.audio.audio_path == "/tmp/test_audio.wav"
        assert "decode_duration_ms" in result.processing_stats

    @pytest.mark.asyncio
    async def test_process_no_audio(self, service):
        """无音频视频 → audio=None, transcript=None"""
        input_result = make_input_result()

        mock_ctx = PipelineContext(
            input_id=input_result.input_id,
            video_path=input_result.temp_path,
            config=service._config,
            _input_metadata=input_result.video_metadata,
            fps=30.0,
            total_frames=300,
            audio_path=None,      # 无音频
            sampled_frames=[
                SampledFrame(
                    frame_index=0,
                    timestamp=0.0,
                    data=np.zeros((480, 640, 3), dtype=np.uint8),
                )
            ],
            stats={},
        )
        mock_ctx.raw_frames = MagicMock()
        mock_ctx.raw_frames.spill_count = 0

        with patch.object(service.pipeline, "run", AsyncMock(return_value=mock_ctx)):
            result = await service.process(input_result)

        assert result.audio is None
        assert result.transcript is None

    @pytest.mark.asyncio
    async def test_process_metadata_passthrough(self, service):
        """metadata 从 InputResult 透传"""
        input_result = make_input_result()

        mock_ctx = PipelineContext(
            input_id=input_result.input_id,
            video_path=input_result.temp_path,
            config=service._config,
            _input_metadata=input_result.video_metadata,
            stats={},
        )
        mock_ctx.raw_frames = MagicMock()
        mock_ctx.raw_frames.spill_count = 0

        with patch.object(service.pipeline, "run", AsyncMock(return_value=mock_ctx)):
            result = await service.process(input_result)

        assert result.metadata is not None
        assert result.metadata.duration == 10.0
        assert result.metadata.width == 1920
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/preprocessing_layer/test_service.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 PreprocessingService**

```python
# src/illegal_review/preprocessing_layer/service.py
import logging
import numpy as np
from src.illegal_review.data_models import (
    InputResult, PreprocessingResult, FrameData, AudioData,
)
from src.illegal_review.preprocessing_layer.pipeline import Pipeline
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.preprocessing_layer.stages.decode_stage import DecodeStage
from src.illegal_review.preprocessing_layer.stages.audio_extract_stage import AudioExtractStage
from src.illegal_review.preprocessing_layer.stages.frame_sample_stage import FrameSampleStage
from src.illegal_review.preprocessing_layer.stages.speech_stage import SpeechStage
from src.illegal_review.preprocessing_layer.stages.ocr_stage import OCRStage
from src.illegal_review.preprocessing_layer.utils.temp_cleaner import TempCleaner
from src.illegal_review.preprocessing_layer.utils.frame_io import encode_frame
from src.illegal_review.config.settings import PreprocessingConfig

logger = logging.getLogger(__name__)


class PreprocessingService:
    """预处理层对外门面"""

    def __init__(self, config: PreprocessingConfig):
        self._config = config
        self.pipeline = Pipeline()
        # 注册 Stage（构造时预加载 Speech/OCR 模型）
        self.pipeline.add_stage(DecodeStage(config))
        self.pipeline.add_stage(AudioExtractStage(config))
        self.pipeline.add_stage(FrameSampleStage(config))
        self.pipeline.add_stage(SpeechStage(config))
        self.pipeline.add_stage(OCRStage(config))

    async def process(self, input_result: InputResult) -> PreprocessingResult:
        ctx = PipelineContext(
            input_id=input_result.input_id,
            video_path=input_result.temp_path,
            config=self._config,
            _input_metadata=input_result.video_metadata,
        )
        async with TempCleaner(ctx):
            ctx = await self.pipeline.run(ctx)
        return self._to_result(ctx)

    def _to_result(self, ctx: PipelineContext) -> PreprocessingResult:
        """边界转换：PipelineContext (np.ndarray) → PreprocessingResult (bytes)"""
        # 帧：np.ndarray → JPEG bytes
        frames = None
        if ctx.sampled_frames:
            frames = [
                FrameData(
                    frame_index=f.frame_index,
                    timestamp=f.timestamp,
                    image_data=encode_frame(f.data, self._config),
                    width=f.data.shape[1],
                    height=f.data.shape[0],
                )
                for f in ctx.sampled_frames
            ]

        # 音频：路径 → AudioData
        audio = None
        if ctx.audio_path:
            audio = AudioData(
                audio_path=ctx.audio_path,
                sample_rate=self._config.audio_sample_rate,
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
                "total_frames": float(ctx.total_frames or 0),
                "sampled_frames": float(len(ctx.sampled_frames) if ctx.sampled_frames else 0),
                "error_count": float(len(ctx.errors)),
            },
        )
```

- [ ] **Step 4: 更新 preprocessing_layer/__init__.py**

```python
# src/illegal_review/preprocessing_layer/__init__.py
"""
预处理层模块

DAG Pipeline 架构：
1. 视频解码 (DecodeStage)：FFmpeg pipe 模式解码为帧序列
2. 音频提取 (AudioExtractStage)：提取音频 → 16kHz 单声道 PCM
3. 帧采样 (FrameSampleStage)：自适应间隔 + 场景变化检测
4. 语音转写 (SpeechStage)：Whisper small 模型
5. 文字识别 (OCRStage)：EasyOCR 中英文
"""

from src.illegal_review.preprocessing_layer.service import PreprocessingService

__all__ = ["PreprocessingService"]
```

- [ ] **Step 5: 运行测试验证通过**

```bash
python -m pytest tests/preprocessing_layer/test_service.py -v
```

Expected: PASS (3/3)

- [ ] **Step 6: Commit**

```bash
git add src/illegal_review/preprocessing_layer/service.py src/illegal_review/preprocessing_layer/__init__.py tests/preprocessing_layer/test_service.py
git commit -m "feat: add PreprocessingService with _to_result boundary conversion"
```

---

### Task 15: 集成测试 — 全部测试运行

- [ ] **Step 1: 运行全部预处理层测试**

```bash
python -m pytest tests/preprocessing_layer/ -v
```

Expected: ALL PASS (所有 14 个测试模块)

- [ ] **Step 2: 验证模块导入**

```bash
python -c "from src.illegal_review.preprocessing_layer import PreprocessingService; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: 运行全项目测试确保无回归**

```bash
python -m pytest tests/ -v
```

Expected: 所有已有输入层测试继续通过，预处理层测试全部通过

- [ ] **Step 4: Commit**

```bash
git add tests/preprocessing_layer/__init__.py
git commit -m "test: add preprocessing layer integration tests — all 15 task modules passing"
```

---

## 依赖关系图

```
Task 1 (data_models + config)
  │
  ├──→ Task 2 (exceptions)
  │      │
  │      └──→ Task 3 (context)
  │             │
  │             └──→ Task 4 (stage)
  │                    │
  │         ┌──────────┼──────────┐
  │         ▼          ▼          ▼
  │    Task 5      Task 6     Task 7
  │  (frame_io) (ffmpeg)  (temp_cleaner)
  │         │          │          │
  │         └──────────┼──────────┘
  │                    │
  │    ┌───────┬───────┼───────┬───────┐
  │    ▼       ▼       ▼       ▼       ▼
  │  Task 8  Task 9  Task 10 Task 11 Task 12
  │ (decode)(audio) (frame) (speech)(ocr)
  │    │       │       │       │       │
  │    └───────┴───────┼───────┴───────┘
  │                    │
  │                    ▼
  │                 Task 13 (pipeline)
  │                    │
  │                    ▼
  │                 Task 14 (service)
  │                    │
  └────────────────────┴──→ Task 15 (integration)
```

**并行机会**：Tasks 2-7 与 Task 1 可部分并行（Task 2 独立）。Tasks 8-12 可并行（各自依赖 Stage 基类）。Tasks 5、6、7 可并行。
