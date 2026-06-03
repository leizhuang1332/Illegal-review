# 层间参数统一对齐设计

**日期：** 2026-06-03
**状态：** 设计完成待实现

---

## 1. 背景与问题

项目采用 5 层架构（输入层 → 预处理层 → 分析引擎层 → 融合决策层 → 输出层），
目前已实现输入层、预处理层、分析引擎层的文本分析引擎。

代码审查发现以下问题：

### 1.1 重复定义（3 组）

| 模型 | 重复位置 | 问题 |
|------|----------|------|
| `VideoMetadata` | `data_models.py` + `input_layer/models.py` | 字段相同，但后者多了 `gt=0` 验证约束 |
| `SourceInfo` | `data_models.py` + `input_layer/models.py` | 完全相同，纯冗余 |
| `InputResult` / `IngestResult` | `data_models.py` + `input_layer/models.py` | 功能等价但字段名不同（`error_message` vs `error`），状态枚举不一致 |

### 1.2 类型不匹配

[preprocessing_layer/service.py:31](src/illegal_review/preprocessing_layer/service.py#L31) 声明参数类型为 `InputResult`（来自 `data_models.py`），
但实际调用方（[integration_test.py:60](scripts/integration_test.py#L60)）传入的是 `IngestResult`（来自 `input_layer/models.py`）。
由于 Python 鸭子类型，目前不崩溃，但无类型安全保障。

### 1.3 拆包传递

文本分析引擎入口 `analyze_all(video_id, ocr_results, transcript)` 逐个接收参数，
而非接收完整的 `PreprocessingResult`，导致：
- 参数增加时需要改接口签名
- 调用方需要了解引擎内部需要哪些字段

---

## 2. 设计目标

1. **消除重复** — 层间传递模型只在 `data_models.py` 定义一份
2. **类型一致** — 上层出参直接作为下层入参，类型签名真实反映运行时
3. **整体传递** — 下层接收上层输出的完整对象，自行提取所需字段
4. **API 层隔离** — `input_layer/models.py` 只保留外部 API 请求/响应模型

---

## 3. 统一模型定义

### 3.1 `InputResult`（合并 `InputResult` + `IngestResult`）

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

| 合并决策 | 采纳来源 | 原因 |
|---------|---------|------|
| `input_type: Literal` | IngestResult | 更严格的类型约束 |
| `status` 含 "processing" | IngestResult | 更完整的状态覆盖 |
| `error` 字段名 | IngestResult | `error_message` 冗余 |
| `Field(description=...)` | InputResult | 支持自动文档生成 |
| `default_factory` on `created_at` | IngestResult | 调用方不必每次填写 |

### 3.2 `VideoMetadata`（合并两处约束）

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

保留 `data_models.py` 的 `description` 文档注释风格，
增加 `input_layer/models.py` 版本的 `gt=0` 验证约束。

### 3.3 `SourceInfo`

保持不变（两处定义完全一致），删除重复。

### 3.4 层间传递关系

```
[输入层]
  │ 输出: InputResult
  ▼
[预处理层]
  │ 输入: InputResult (整体传入)
  │ 输出: PreprocessingResult
  ▼
[分析引擎层 — 文本分析]
  │ 输入: PreprocessingResult (整体传入)
  │ 输出: TextAnalysisResult
  ▼
[融合决策层] (待实现)
  │ 输入: TextAnalysisResult + ImageAnalysisResult + AudioAnalysisResult
  │      + RuleEngineResult + AIEngineResult
  │ 输出: FinalDecision
  ▼
[输出层] (待实现)
  │ 输入: FinalDecision
```

---

## 4. 具体变更清单

### 4.1 `data_models.py`
- 合并 `InputResult`（原第 40 行）与 `IngestResult` → 统一 `InputResult`
- `VideoMetadata` 增加 `gt=0` 验证约束
- 删除原 `InputResult` 定义（被合并版本替代）

### 4.2 `input_layer/models.py`
- 删除 `VideoMetadata`（第 46-54 行）、`SourceInfo`（第 57-61 行）、`IngestResult`（第 64-74 行）
- 改为从 `data_models` 引用：
  ```python
  from src.illegal_review.data_models import VideoMetadata, SourceInfo, InputResult
  ```
- 保留纯 API 模型：`UploadOptions`, `VideoUploadRequest`, `UrlFetchRequest`,
  `LiveStreamRequest`, `ChunkedUploadCreateRequest`, `TaskStatusResponse`,
  `ChunkedUploadCreateResponse`, `ErrorResponse`

### 4.3 `input_layer/__init__.py`
- 导入来源从 `input_layer.models` 改为 `data_models`

### 4.4 `input_layer/service.py`
- 导入 `InputResult` 改为 `data_models.InputResult`
- `IngestResult` 引用全部替换为 `InputResult`

### 4.5 `preprocessing_layer/service.py`
- **无需改动** — 已正确导入 `data_models.InputResult`
- 统一后调用方传入的也是统一 `InputResult`，类型自动匹配

### 4.6 `analysis_engine_layer/text_analysis/service.py`
- 接口签名变更：
  ```python
  # 旧：analyze_all(self, video_id, ocr_results=None, transcript=None)
  # 新：analyze_all(self, prep_result: PreprocessingResult)
  ```
- 内部从 `prep_result` 提取 `input_id`、`ocr_results`、`transcript`

### 4.7 `scripts/integration_test.py`
- `run_with_video()` — `input_result` 自动适配统一 `InputResult`，无需改动
- `run_mock()` — 调用方式更新为 `analyze_all(prep_result)` 形式
- `run_mock_preprocess()` — 直接使用 `data_models.InputResult` 构造模拟数据

---

## 5. 未变更文件

| 文件 | 原因 |
|------|------|
| `config/settings.py` | 配置与数据模型无关 |
| `preprocessing_layer/context.py` | 内部管线上下文，不跨层 |
| `text_analysis/analyzer.py` 等 15 个内部模块 | 内部逻辑，接口不变 |
| `stages/*.py` | 操作 `PipelineContext`，不直接接触层模型 |
| `cli.py` | 暂只做命令分发 |
| `router.py` | 只引用 API 模型 |

---

## 6. 未来层扩展指引

后续实现各引擎时，按此模式：

1. **输入模型** = 上层输出的完整对象（不定义独立输入类）
2. **输出模型** 定义在 `data_models.py` 中
3. 引擎内部自行提取所需字段

```
图像识别引擎: process(prep_result: PreprocessingResult) → ImageAnalysisResult
音频分析引擎: process(prep_result: PreprocessingResult) → AudioAnalysisResult
规则引擎:     evaluate(analysis_results: ...) → RuleEngineResult
AI引擎:       predict(analysis_results: ...) → AIEngineResult
融合决策:     decide(rule_result, ai_result, ...) → FinalDecision
```

---

## 7. 验证标准

实现后验证以下场景：

1. **类型安全**: 所有 `isinstance` 检查和 IDE 类型提示正确
2. **现有测试通过**: `pytest tests/` 全部通过
3. **集成测试通过**: `python scripts/integration_test.py` 正常运行
4. **全流程通**: 文件上传 → 预处理 → 文本分析 三条路径（视频、模拟、mock-preprocess）均正常
