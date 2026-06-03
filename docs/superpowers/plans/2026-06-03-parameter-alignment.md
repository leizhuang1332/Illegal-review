# 层间参数统一对齐 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 `data_models.py` 与 `input_layer/models.py` 之间 3 组重复模型类，统一层间参数传递方式（上层出参 = 下层入参）。

**Architecture:** 方案 A — 统一模型 + 整体传递。`data_models.py` 作为层间模型唯一来源，`input_layer/models.py` 只保留 API 请求/响应模型。文本分析引擎入口改为接收 `PreprocessingResult` 整体对象。

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI

---

## Scope Check

本次变更仅涉及**已实现层**的参数对齐，不涉及未实现层（图像/音频/规则/AI 引擎、融合决策层、输出层）。按文件划分 5 个独立任务，各任务互不阻塞。

## 文件结构总览

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/illegal_review/data_models.py` | 修改 | 合并 `InputResult` + `IngestResult`；增强 `VideoMetadata` |
| `src/illegal_review/input_layer/models.py` | 修改 | 删除 `VideoMetadata`, `SourceInfo`, `IngestResult`；改为从 `data_models` 引用 |
| `src/illegal_review/input_layer/__init__.py` | 修改 | 更新导出路径 |
| `src/illegal_review/input_layer/service.py` | 修改 | 导入改用 `data_models.InputResult` |
| `src/illegal_review/input_layer/metadata.py` | 修改 | 导入 `VideoMetadata` 来源改为 `data_models` |
| `src/illegal_review/analysis_engine_layer/text_analysis/service.py` | 修改 | `analyze_all()` 签名改为接收 `PreprocessingResult` |
| `scripts/integration_test.py` | 修改 | 更新 `analyze_all()` 调用方式 |
| `scripts/verify_text_analysis.py` | 修改 | 更新 `analyze_all()` 调用方式 |
| `tests/analysis_engine_layer/text_analysis/test_service.py` | 修改 | 更新 `analyze_all()` 调用方式 |
| `src/illegal_review/preprocessing_layer/service.py` | 无需改动 | 已正确导入 `data_models.InputResult` |

---

### Task 1: 统一 `data_models.py` 中的模型

**Files:**
- Modify: `src/illegal_review/data_models.py` — 修改 `InputResult`（合并 `IngestResult` 的特性）、增强 `VideoMetadata`

- [ ] **Step 1: 修改 `VideoMetadata` — 增加 `gt=0` 验证约束**

将 `input_layer/models.py` 版本中的 `gt=0` 验证合并到 `data_models.py`：

```python
class VideoMetadata(BaseModel):
    """视频元数据"""
    duration: float = Field(gt=0, description="视频时长（秒）")
    fps: float = Field(gt=0, description="帧率")
    width: int = Field(gt=0, description="宽度（像素）")
    height: int = Field(gt=0, description="高度（像素）")
    codec: str = Field(description="视频编码格式")
    audio_codec: Optional[str] = Field(default=None, description="音频编码格式")
    bitrate: Optional[int] = Field(default=None, description="比特率（bps）")
```

只需将 `duration`, `fps`, `width`, `height` 四个字段加上 `gt=0`。

- [ ] **Step 2: 统一 `InputResult` — 合并 `IngestResult` 的特性**

替换原有的 `InputResult` 定义：

```python
class InputResult(BaseModel):
    """输入层输出结果 = 预处理层输入"""
    input_id: UUID = Field(description="唯一标识")
    input_type: Literal["file", "url", "live"] = Field(description="输入类型")
    source_info: SourceInfo = Field(description="源信息")
    video_metadata: Optional[VideoMetadata] = Field(default=None, description="视频元数据")
    temp_path: str = Field(description="临时文件路径")
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        default="pending", description="任务状态"
    )
    error: Optional[str] = Field(default=None, description="错误信息")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="创建时间"
    )
    processed_at: Optional[datetime] = Field(default=None, description="处理完成时间")
```

合并决策：
- `input_type` 从 `str` → `Literal["file", "url", "live"]`（来自 IngestResult）
- `status` 增加 `"processing"` 枚举值（来自 IngestResult）
- `error_message` → `error`（来自 IngestResult，更简洁）
- `created_at` 加 `default_factory`（来自 IngestResult，调用方不必每次填写）
- 保留全部 `Field(description=...)`（来自原 InputResult）

需要添加导入：`from typing import Literal`
需要修改导入：`from datetime import datetime` → `from datetime import datetime, timezone`（因为使用了 `timezone.utc`）

- [ ] **Step 3: 确认合并结果**

目视检查 `data_models.py` 中不再有 `IngestResult` 类（本来就只在 `input_layer/models.py` 中定义），
且统一后的 `InputResult` 字段包含了原本两个类的所有必要字段。

Run: `python -c "from src.illegal_review.data_models import InputResult; print(InputResult.model_json_schema())"`
Expected: 无导入错误，输出包含所有字段的 JSON Schema

---

### Task 2: 清理 `input_layer/models.py` 中的重复定义

**Files:**
- Modify: `src/illegal_review/input_layer/models.py` — 删除 `VideoMetadata`, `SourceInfo`, `IngestResult`
- Modify: `src/illegal_review/input_layer/__init__.py` — 更新导出路径
- Modify: `src/illegal_review/input_layer/service.py` — 导入改用 `data_models.InputResult`
- Modify: `src/illegal_review/input_layer/metadata.py` — `VideoMetadata` 导入来源改为 `data_models`

- [ ] **Step 1: 从 `input_layer/models.py` 中删除重复类**

删除以下 3 个类定义：
- `VideoMetadata` 类（第 46-54 行）
- `SourceInfo` 类（第 57-61 行）
- `IngestResult` 类（第 64-74 行）

在文件顶部添加引用导入：

```python
from src.illegal_review.data_models import VideoMetadata, SourceInfo, InputResult
```

文件中保留的 API 模型（不动）：`UploadOptions`, `VideoUploadRequest`, `UrlFetchRequest`,
`LiveStreamRequest`, `ChunkedUploadCreateRequest`, `TaskStatusResponse`,
`ChunkedUploadCreateResponse`, `ErrorResponse`

- [ ] **Step 2: 更新 `input_layer/__init__.py` 的导出**

```python
# 旧（第 15 行）：
from src.illegal_review.input_layer.models import IngestResult, VideoMetadata, SourceInfo

# 新：
from src.illegal_review.data_models import InputResult, VideoMetadata, SourceInfo
```

- [ ] **Step 3: 更新 `input_layer/service.py` 的导入**

```python
# 旧（第 7 行）：
from src.illegal_review.input_layer.models import IngestResult, SourceInfo

# 新：
from src.illegal_review.data_models import InputResult, SourceInfo
```

然后将文件中所有的 `IngestResult` 引用替换为 `InputResult`：
- 第 17 行：`self._tasks: Dict[UUID, IngestResult] = {}`
- 第 27 行：`async def handle_file_upload(...) -> IngestResult:`
- 第 29 行：`result = IngestResult(...)`
- 第 95 行：`async def get_task_status(...) -> Optional[IngestResult]:`

全部替换为 `InputResult`。

- [ ] **Step 4: 更新 `input_layer/metadata.py` 的导入**

```python
# 旧（第 11 行）：
from src.illegal_review.input_layer.models import VideoMetadata

# 新：
from src.illegal_review.data_models import VideoMetadata
```

- [ ] **Step 5: 验证无导入错误**

Run: `python -c "from src.illegal_review.input_layer.service import InputService; print('OK')"`
Expected: 打印 "OK"，无 `ImportError`

---

### Task 3: 修改文本分析引擎入口签名

**Files:**
- Modify: `src/illegal_review/analysis_engine_layer/text_analysis/service.py`

- [ ] **Step 1: 修改 `analyze_all` 签名及实现**

```python
# 旧：
async def analyze_all(
    self,
    video_id: UUID,
    ocr_results: Optional[List[OCRResult]] = None,
    transcript: Optional[str] = None,
) -> TextAnalysisResult:

# 新：
async def analyze_all(
    self,
    prep_result: PreprocessingResult,
) -> TextAnalysisResult:
```

同时更新实现体——从 `prep_result` 提取字段：

```python
async def analyze_all(
    self,
    prep_result: PreprocessingResult,
) -> TextAnalysisResult:
    """完整分析：从预处理结果中提取 OCR + 语音来源"""
    tasks = {}

    if prep_result.ocr_results:
        ocr_text = TextAdaptor.extract_ocr_text(
            prep_result.ocr_results,
            min_confidence=self._config.ocr_confidence_threshold,
            max_length=self._config.ocr_max_text_length,
        )
        if ocr_text:
            tasks["ocr"] = self._analyzer.analyze(ocr_text, source="ocr")

    if prep_result.transcript:
        trans_text = TextAdaptor.extract_transcript(
            prep_result.transcript,
            max_length=self._config.max_text_length,
        )
        if trans_text:
            tasks["transcript"] = self._analyzer.analyze(trans_text, source="transcript")

    results = {}
    if tasks:
        for name, task in tasks.items():
            results[name] = await task

    return self._merger.merge(
        ocr_result=results.get("ocr"),
        transcript_result=results.get("transcript"),
        video_id=prep_result.input_id,
    )
```

修改 `from src.illegal_review.data_models import (...)` 中增加 `PreprocessingResult`：
```python
from src.illegal_review.data_models import (
    PreprocessingResult, TextAnalysisResult,
)
```
（移除了 `OCRResult`，因为它不再在签名中使用；`PreprocessingResult` 内部包含了它）

移除不再需要的导入：

- `from uuid import UUID`（不再直接作为参数）
- `from typing import List, Optional`（签名中不再使用）

- [ ] **Step 2: 验证无语法错误**

Run: `python -c "from src.illegal_review.analysis_engine_layer.text_analysis.service import TextAnalysisService; print('OK')"`
Expected: 打印 "OK"

---

### Task 4: 更新所有调用方

**Files:**
- Modify: `scripts/integration_test.py`
- Modify: `scripts/verify_text_analysis.py`
- Modify: `tests/analysis_engine_layer/text_analysis/test_service.py`

- [ ] **Step 1: 更新 `scripts/integration_test.py`**

**函数 `run_with_video()`（第 76-80 行）：**
```python
# 旧：
text_result = await text_service.analyze_all(
    video_id=preprocess_result.input_id,
    ocr_results=preprocess_result.ocr_results,
    transcript=preprocess_result.transcript,
)

# 新：
text_result = await text_service.analyze_all(preprocess_result)
```

**函数 `run_mock()`（第 110-118 行）：**
需要构造一个 `PreprocessingResult` 来传入：

```python
from src.illegal_review.data_models import PreprocessingResult, OCRResult, TextAnalysisResult

# 构造模拟 PreprocessingResult
prep_result = PreprocessingResult(
    input_id=uuid4(),
    ocr_results=[
        OCRResult(text="免费领取毒品", confidence=0.95, bbox=None, frame_index=0),
        OCRResult(text="联系微信", confidence=0.80, bbox=None, frame_index=1),
        OCRResult(text="今晚一起打游戏", confidence=0.90, bbox=None, frame_index=2),
    ],
    transcript="我非常愤怒，这个平台全是骗人的，大家千万不要上当",
)
result = await service.analyze_all(prep_result)
```

**函数 `run_mock_preprocess()`（第 131-176 行）：**
```python
# 旧：
text_result = await text_service.analyze_all(
    video_id=preprocess_result.input_id,
    ocr_results=preprocess_result.ocr_results,
    transcript=preprocess_result.transcript,
)

# 新：
text_result = await text_service.analyze_all(preprocess_result)
```

- [ ] **Step 2: 更新 `scripts/verify_text_analysis.py`**

每个 `analyze_all` 调用（第 57、80、98 行）改为构造 `PreprocessingResult` 传入：

```python
# 示例：场景 A
from src.illegal_review.data_models import PreprocessingResult

prep_result = PreprocessingResult(
    input_id=uuid4(),
    ocr_results=[
        OCRResult(text="免费领取毒品", confidence=0.95, bbox=None, frame_index=0),
        OCRResult(text="联系客服微信xxx", confidence=0.80, bbox=None, frame_index=1),
    ],
    transcript="今天我们来讨论一下正常的内容审核流程和技术方案",
)
result_a = await service.analyze_all(prep_result)
```

场景 B（第 80 行）和场景 C（第 98 行）同理构造对应的 `PreprocessingResult`。

添加导入：
```python
from src.illegal_review.data_models import OCRResult, TextCategory, PreprocessingResult
```

- [ ] **Step 3: 更新 `tests/analysis_engine_layer/text_analysis/test_service.py`**

```python
# 第 41 行——旧：
result = await service.analyze_all(video_id, ocr_results, transcript)

# 新：
from src.illegal_review.data_models import PreprocessingResult
prep_result = PreprocessingResult(
    input_id=video_id,
    ocr_results=ocr_results,
    transcript=transcript,
)
result = await service.analyze_all(prep_result)
```

第 52 行同理：
```python
# 旧：
result = await service.analyze_all(video_id, ocr_results=None, transcript=None)

# 新：
prep_result = PreprocessingResult(input_id=video_id, ocr_results=None, transcript=None)
result = await service.analyze_all(prep_result)
```

---

### Task 5: 运行测试验证

- [ ] **Step 1: 运行全部单元测试**

Run: `cd d:/TraeProject/Illegal-review && python -m pytest tests/ -v --tb=short 2>&1`
Expected: 全部通过（至少包含 input_layer 和 text_analysis 测试）

- [ ] **Step 2: 运行集成测试（模拟模式）**

Run: `cd d:/TraeProject/Illegal-review && python scripts/integration_test.py 2>&1`
Expected: 打印文本分析引擎验证结果，无报错

- [ ] **Step 3: 运行集成测试（mock-preprocess 模式）**

Run: `cd d:/TraeProject/Illegal-review && python scripts/integration_test.py --mock-preprocess 2>&1`
Expected: 打印层间数据流验证结果，无报错

- [ ] **Step 4: 运行文本分析引擎验证脚本**

Run: `cd d:/TraeProject/Illegal-review && python scripts/verify_text_analysis.py 2>&1`
Expected: 三个场景（A/B/C）全部验证完成
