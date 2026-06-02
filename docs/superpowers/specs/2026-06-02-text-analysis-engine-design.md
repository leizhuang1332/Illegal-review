# 文本分析引擎设计文档

> **视频违规审核系统 — 分析引擎层·文本分析引擎**
> 状态: 已批准 | 日期: 2026-06-02 | 审查: 2026-06-02

---

## 1. 概述

文本分析引擎是分析引擎层的第一个子引擎，负责处理视频内容中的文本信息（包括 OCR 识别文本和语音转写文本），提供语义编码、敏感词检测、情感分析、实体识别和文本分类能力。

### 1.1 引擎定位

```
上游 (预处理层)                    下游 (分析引擎层)
PreprocessingResult ──→ TextAnalysisService ──→ TextAnalysisResult
  ├─ ocr_results[]                            ├─ OCR分析结果
  ├─ transcript                               ├─ 语音分析结果
  └─ transcript_segments[]                    ├─ 违规检测汇总
                                               └─ 处理统计
```

- **上游依赖**：预处理层输出的 `PreprocessingResult`（OCR 文本 + 语音转写文本）
- **下游输出**：`TextAnalysisResult` → 规则引擎、AI 引擎
- **职责边界**：专注于文本级别的特征提取和违规检测，不涉及图像、音频分析

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **模块化管道** | 每个分析能力封装为独立模块，可插拔、可替换 |
| **来源分离** | OCR 文本和语音转写文本分别分析，保留来源标识 |
| **并行执行** | 互不依赖的分析模块通过 `asyncio.gather` 并行 |
| **可扩展分类** | 分类类别使用 `str`+`Enum`，新增类别不改业务代码 |
| **轻量优先** | 情感分析首版使用词典法，避免额外模型加载 |
| **构造预加载** | 模型在 Service 构造时加载，避免首次请求延迟 |

---

## 2. 架构设计

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    TextAnalysisService                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │  OCR Input Adaptor  │    │  Transcript Adaptor │             │
│  │ (从OCRResult提取)   │    │ (从transcript提取)  │             │
│  └─────────┬───────────┘    └──────────┬──────────┘             │
│            │                           │                        │
│            └──────────┬────────────────┘                        │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  TextAnalyzer (核心分析器)                 │   │
│  │                                                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │   │
│  │  │预处理    │→ │语义编码  │→ │敏感词检  │→ │情感分析   │  │   │
│  │  │(清洗/分词)│  │(BERT)   │  │测(AC自动机)│  │(Sentiment)│  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └───────────┘  │   │
│  │                                       │                 │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┘                 │   │
│  │  │NER实体  │← │文本分类  │← ┘                           │   │
│  │  │识别     │  │(6分类)   │                              │   │
│  │  └─────────┘  └─────────┘                              │   │
│  └──────────────────────────────┬──────────────────────────┘   │
│                                 │                              │
│                                 ▼                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               ResultMerger (结果合并器)                   │   │
│  │  OCR结果 + 语音结果 → TextAnalysisResult                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 执行流程

```
1. 用户调用 service.analyze_all(video_id, ocr_results, transcript)
2. Adaptor 提取 OCR 文本和语音文本（自动处理 None）
3. 两个文本分别送入 TextAnalyzer.analyze()
4. TextAnalyzer 内部：预处理 → 5个模块并行执行
5. 返回 SourceAnalysis（含语义向量、敏感词、情感、实体、分类)
6. ResultMerger 合并为 TextAnalysisResult
7. 违规检测汇总：敏感词 + 非 normal 分类结果
```

### 2.3 文件结构

```
src/illegal_review/analysis_engine_layer/text_analysis/
├── __init__.py              # 导出 TextAnalysisService
├── service.py               # TextAnalysisService 门面
├── models.py                # 内部数据模型
├── analyzer.py              # TextAnalyzer 核心分析器
├── merger.py                # ResultMerger 结果合并
├── adaptor.py               # 输入适配器
├── preprocessor.py          # 文本预处理
├── semantic.py              # BERT 语义编码模块
├── sensitive_matcher.py     # AC自动机敏感词匹配
├── sentiment.py             # 情感分析模块
├── ner.py                   # 命名实体识别模块
├── classifier.py            # 文本分类器
└── exceptions.py            # 自定义异常

tests/preprocessing_layer/text_analysis/
├── __init__.py
├── test_preprocessor.py
├── test_semantic.py
├── test_sensitive_matcher.py
├── test_sentiment.py
├── test_ner.py
├── test_classifier.py
├── test_analyzer.py
├── test_service.py
├── test_merger.py
├── test_adaptor.py
└── test_exceptions.py
```

---

## 3. 数据模型

### 3.1 TextAnalysisResult（修改）

```python
class TextCategory(str, Enum):
    """文本分类类别（str+Enum = 可序列化为纯字符串，可扩展）"""
    PORN = "porn"
    VIOLENCE = "violence"
    POLITICAL = "political"
    AD = "ad"
    COPYRIGHT = "copyright"
    NORMAL = "normal"


class TextAnalysisResult(BaseModel):
    """文本分析结果"""
    video_id: UUID = Field(description="视频标识")
    ocr: Optional["SourceAnalysis"] = Field(default=None, description="OCR文本分析结果")
    transcript: Optional["SourceAnalysis"] = Field(default=None, description="语音转写分析结果")
    violations: List[ViolationDetection] = Field(default_factory=list, description="违规检测汇总")
    processing_stats: Dict[str, float] = Field(default_factory=dict, description="处理统计")


class SourceAnalysis(BaseModel):
    """单来源文本分析结果"""
    source: str = Field(description="来源：ocr / transcript")
    text_length: int = Field(description="文本长度")
    semantic_embedding: Optional[List[float]] = Field(default=None, description="语义嵌入 (768维)")
    sensitive_words: List[SensitiveWord] = Field(default_factory=list, description="敏感词列表")
    sentiment_score: Optional[float] = Field(default=None, description="情感分数 -1~1")
    entities: List[Entity] = Field(default_factory=list, description="实体列表")
    category: Optional[CategoryResult] = Field(default=None, description="分类结果")


class SensitiveWord(BaseModel):
    """敏感词匹配结果"""
    word: str = Field(description="敏感词")
    start_pos: int = Field(description="起始位置")
    end_pos: int = Field(description="结束位置")
    match_type: str = Field(description="匹配方式：exact / fuzzy / regex")
    category: str = Field(description="敏感词类别")


class Entity(BaseModel):
    """命名实体"""
    name: str = Field(description="实体名称")
    type: str = Field(description="类型：person / location / organization / time / other")
    start_pos: int = Field(description="起始位置")
    end_pos: int = Field(description="结束位置")
    confidence: float = Field(description="置信度 0~1")


class CategoryResult(BaseModel):
    """文本分类结果"""
    category: TextCategory = Field(description="违规类别")
    confidence: float = Field(description="置信度 0~1")
    scores: Dict[str, float] = Field(description="各类别概率分布")
```

### 3.2 存在但需修改的数据模型

`data_models.py` 中的 `TextAnalysisResult` 需替换为上方定义，并新增 `SourceAnalysis`、`SensitiveWord`、`Entity`、`CategoryResult` 模型。

---

## 4. 配置模型

### 4.1 TextAnalysisConfig（修改）

```python
@dataclass
class TextAnalysisConfig:
    """文本分析引擎配置"""

    # 模型
    bert_model: str = "bert-base-chinese"
    classifier_model_path: Optional[str] = None

    # 敏感词
    sensitive_word_list_path: str = "config/sensitive_words.txt"
    sensitive_fuzzy_match_enabled: bool = True
    sensitive_fuzzy_threshold: float = 0.8

    # 分析功能开关
    sentiment_analysis_enabled: bool = True
    ner_enabled: bool = True

    # 分类
    categories: List[str] = field(
        default_factory=lambda: [c.value for c in TextCategory]
    )
    classifier_threshold: float = 0.5

    # OCR 输入
    ocr_confidence_threshold: float = 0.5
    ocr_max_text_length: int = 5000

    # 批处理
    max_text_length: int = 10000
```

---

## 5. 模块详细设计

### 5.1 文本预处理模块（preprocessor.py）

**职责**：对原始文本进行清洗和标准化，供所有下游模块使用

```python
class TextPreprocessor:
    """文本清洗与分词（使用 jieba，无额外模型依赖）"""

    def __init__(self, language: str = "zh"):
        import jieba
        self._jieba = jieba

    def clean(self, text: str) -> str:
        """去除HTML标签、特殊符号、多余空白"""
        text = re.sub(r'<[^>]+>', '', text)        # HTML 标签
        text = re.sub(r'[^\w\s一-鿿]', ' ', text)  # 特殊符号
        text = re.sub(r'\s+', ' ', text).strip()   # 多余空白
        return text

    def segment(self, text: str) -> List[str]:
        """分词"""
        return self._jieba.lcut(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """去停用词"""
        stopwords = self._get_stopwords()
        return [t for t in tokens if t not in stopwords]

    def normalize(self, text: str) -> str:
        """繁简转换、全角半角统一 — 使用 opencc"""
        import opencc
        converter = opencc.OpenCC('t2s.json')  # 繁体→简体
        text = converter.convert(text)
        # 全角→半角
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    def process(self, text: str) -> PreprocessedText:
        """全流程：clean → normalize → segment → remove_stopwords"""
        text = self.clean(text)
        text = self.normalize(text)
        tokens = self.segment(text)
        tokens = self.remove_stopwords(tokens)
        return PreprocessedText(text=text, tokens=tokens)
```

**关键设计**：
- 停用词表以类级别缓存加载，避免重复读盘
- `process()` 返回 `PreprocessedText`（含 `text` 和 `tokens`），下游按需取用
- 无外部模型依赖，启动快

### 5.2 BERT 语义编码模块（semantic.py）

**职责**：文本 → 768维语义向量

```python
class SemanticEncoder:
    """BERT 语义编码器"""

    def __init__(self, config: TextAnalysisConfig):
        from transformers import AutoTokenizer, AutoModel
        self._tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self._model = AutoModel.from_pretrained(config.bert_model)
        self._model.eval()

    async def encode(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._encode_sync, text)

    def _encode_sync(self, text: str) -> List[float]:
        inputs = self._tokenizer(text, return_tensors="pt",
                                 max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        return embedding
```

**关键设计**：
- 取 `[CLS]` 向量（768维）作为句子嵌入
- 构造时预加载模型，推理通过 `asyncio.to_thread`
- 截断 512 token（BERT 最大输入长度）

### 5.3 敏感词检测模块（sensitive_matcher.py）

**职责**：AC自动机精确匹配 + 模糊匹配

```python
class SensitiveMatcher:
    """AC自动机敏感词匹配"""

    def __init__(self, config: TextAnalysisConfig):
        self._words: Dict[str, str] = {}
        self._fuzzy_enabled = config.sensitive_fuzzy_match_enabled
        self._fuzzy_threshold = config.sensitive_fuzzy_threshold
        self._build_automaton(config.sensitive_word_list_path)

    def _build_automaton(self, path: str):
        """加载敏感词列表 → 构建 AC 自动机"""
        # 词表格式：word,category 每行
        # 构建 trie + fail 指针
        ...

    def match(self, text: str) -> List[SensitiveWord]:
        """精确匹配（AC自动机一次扫描，O(n+m)）"""
        ...

    def fuzzy_match(self, text: str) -> List[SensitiveWord]:
        """模糊匹配（编辑距离/拼音相似度）"""
        ...

    def match_all(self, text: str) -> List[SensitiveWord]:
        """精确 + 模糊匹配，结果合并去重"""
        results = self.match(text)
        if self._fuzzy_enabled:
            results.extend(self.fuzzy_match(text))
        return self._deduplicate(results)
```

**关键设计**：
- 核心用 AC 自动机，一次扫描 O(n + m) 找出所有精确匹配
- 模糊匹配对未命中精确匹配的文本片段做编辑距离计算
- `match_all()` 结果按置信度降序排列并去重

### 5.4 情感分析模块（sentiment.py）

**职责**：判断文本情感倾向（首版词典法）

```python
class SentimentAnalyzer:
    """情感分析 — 基于 lexicon + 规则（无模型依赖）"""

    def __init__(self, config: TextAnalysisConfig):
        self._load_lexicons()

    def analyze(self, text: str) -> float:
        """返回 -1.0 ~ 1.0，正值=正面，负值=负面，0=中性"""
        score = 0.0
        words = jieba.lcut(text)
        for i, w in enumerate(words):
            if w in self._positive_words:
                score += self._apply_weight(i, words) * 1.0
            elif w in self._negative_words:
                score -= self._apply_weight(i, words) * 1.0
        return max(-1.0, min(1.0, score / max(len(words), 1)))
```

**关键设计**：
- 首版使用词典+规则法，无模型加载，启动快
- 支持否定词反转（"不喜欢" → 极性翻转）
- 支持程度副词加权（"非常喜欢" → 强度增强）
- 后续可无侵入替换为 BERT 微调模型

### 5.5 实体识别模块（ner.py）

**职责**：识别命名实体（人物、地点、组织、时间）

```python
class NERecognizer:
    """命名实体识别 — 使用 ckiplab/bert-base-chinese-ner"""

    def __init__(self, config: TextAnalysisConfig):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self._model = AutoModelForTokenClassification.from_pretrained(
            "ckiplab/bert-base-chinese-ner"
        )
        self._model.eval()

    async def recognize(self, text: str) -> List[Entity]:
        return await asyncio.to_thread(self._recognize_sync, text)

    def _recognize_sync(self, text: str) -> List[Entity]:
        # Tokenize → 推理 → BIO 标签解码 → Entity 列表
        ...
```

**关键设计**：
- 使用 `ckiplab/bert-base-chinese-ner`，支持 PER/LOC/ORG/TIME
- 构造时预加载，推理走线程池
- 备选方案：`spacy zh_core_web_sm`（~50MB，更轻量）

### 5.6 文本分类器（classifier.py）

**职责**：6类违规文本分类（可扩展）

```python
class TextClassifier:
    """文本违规分类器 — bert + 分类头"""

    def __init__(self, config: TextAnalysisConfig):
        self._categories = config.categories
        self._threshold = config.classifier_threshold
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = config.classifier_model_path or config.bert_model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self._categories)
        )
        self._model.eval()

    async def classify(self, text: str) -> CategoryResult:
        return await asyncio.to_thread(self._classify_sync, text)

    def _classify_sync(self, text: str) -> CategoryResult:
        inputs = self._tokenizer(text, return_tensors="pt",
                                 max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        scores = {cat: float(probs[i]) for i, cat in enumerate(self._categories)}
        pred_idx = int(probs.argmax())
        return CategoryResult(
            category=self._categories[pred_idx],
            confidence=float(probs[pred_idx]),
            scores=scores,
        )
```

**关键设计**：
- 使用 `AutoModelForSequenceClassification`，`num_labels` 由配置动态决定
- 新类别 → 改 `TextCategory` 枚举 + 更新配置 + 模型微调
- 模型构造时预加载（可指定独立微调模型，或复用 BERT 加分类头）

### 5.7 TextAnalyzer（分析器）

```python
class TextAnalyzer:
    """文本分析器 — 串联所有模块，并行执行互不依赖的分析"""

    def __init__(self, config: TextAnalysisConfig):
        self._preprocessor = TextPreprocessor()
        self._semantic = SemanticEncoder(config)
        self._sensitive = SensitiveMatcher(config)
        self._sentiment = SentimentAnalyzer(config)
        self._ner = NERecognizer(config)
        self._classifier = TextClassifier(config)

    async def analyze(self, text: str, source: str) -> SourceAnalysis:
        cleaned = self._preprocessor.process(text)

        # 并行执行，单个模块失败不阻断其他
        results = await asyncio.gather(
            self._semantic.encode(cleaned.text),
            self._sensitive.match_all(cleaned.text),
            self._sentiment.analyze(cleaned.text),
            self._ner.recognize(cleaned.text),
            self._classifier.classify(cleaned.text),
            return_exceptions=True,
        )

        embed_result, sensitive_result, sentiment_result, ner_result, classify_result = results
        errors: List[str] = []

        embedding = self._unwrap(embed_result, errors, "semantic")
        sensitive_words = self._unwrap(sensitive_result, errors, "sensitive") or []
        sentiment_score = self._unwrap(sentiment_result, errors, "sentiment")
        entities = self._unwrap(ner_result, errors, "ner") or []
        category = self._unwrap(classify_result, errors, "classification")

        return SourceAnalysis(
            source=source,
            text_length=len(text),
            semantic_embedding=embedding,
            sensitive_words=sensitive_words,
            sentiment_score=sentiment_score,
            entities=entities,
            category=category,
            errors=errors,
        )

    def _unwrap(self, result, errors, module):
        if isinstance(result, Exception):
            errors.append(f"{module}: {result}")
            return None
        return result
```

### 5.8 Service 层（service.py）

```python
class TextAnalysisService:
    """文本分析引擎对外门面"""

    def __init__(self, config: TextAnalysisConfig):
        self._config = config
        self._analyzer = TextAnalyzer(config)
        self._merger = ResultMerger()
        # 构造时预加载所有模型

    async def analyze_all(self,
                          video_id: UUID,
                          ocr_results: Optional[List[OCRResult]] = None,
                          transcript: Optional[str] = None) -> TextAnalysisResult:
        """完整分析：OCR + 语音，自动跳过 None"""

        tasks = {}
        if ocr_results:
            text = TextAdaptor.extract_ocr_text(
                ocr_results, self._config.ocr_confidence_threshold
            )
            if text:
                tasks["ocr"] = self._analyzer.analyze(text, source="ocr")

        if transcript:
            text = TextAdaptor.extract_transcript(transcript)
            if text:
                tasks["transcript"] = self._analyzer.analyze(text, source="transcript")

        results = {}
        if tasks:
            for name, task in tasks.items():
                results[name] = await task

        return self._merger.merge(
            ocr_result=results.get("ocr"),
            transcript_result=results.get("transcript"),
        )
```

### 5.9 结果合并器（merger.py）

```python
class ResultMerger:
    """合并 OCR 和语音的分析结果"""

    def merge(self,
              ocr_result: Optional[SourceAnalysis],
              transcript_result: Optional[SourceAnalysis],
              video_id: UUID) -> TextAnalysisResult:

        violations: List[ViolationDetection] = []
        for src in [ocr_result, transcript_result]:
            if src is None:
                continue
            # 敏感词 → 违规证据
            for sw in src.sensitive_words:
                violations.append(ViolationDetection(
                    type="sensitive_word",
                    category=sw.category,
                    confidence=1.0,
                    evidence={"source": src.source, "word": sw.word},
                ))
            # 非 normal 分类 → 违规证据
            if src.category and src.category.category != TextCategory.NORMAL:
                violations.append(ViolationDetection(
                    type="text_classification",
                    category=src.category.category.value,
                    confidence=src.category.confidence,
                    evidence={"source": src.source, "scores": src.category.scores},
                ))

        return TextAnalysisResult(
            video_id=video_id,
            ocr=ocr_result,
            transcript=transcript_result,
            violations=self._deduplicate(violations),
            processing_stats={"violations_count": float(len(violations))},
        )

    def _deduplicate(self, violations: List[ViolationDetection]) -> List[ViolationDetection]:
        """相同 category 取最高置信度"""
        seen = {}
        for v in violations:
            key = (v.type, v.category)
            if key not in seen or v.confidence > seen[key].confidence:
                seen[key] = v
        return list(seen.values())
```

**注意**：Service 层的 `analyze_all()` 调用时传入 `video_id`：

```python
return self._merger.merge(
    ocr_result=results.get("ocr"),
    transcript_result=results.get("transcript"),
    video_id=video_id,
)
```

### 5.10 输入适配器（adaptor.py）

```python
class TextAdaptor:
    """将 PreprocessingResult 中的文本提取为标准字符串"""

    @staticmethod
    def extract_ocr_text(ocr_results: List[OCRResult],
                          min_confidence: float = 0.5,
                          max_length: int = 5000) -> Optional[str]:
        if not ocr_results:
            return None
        valid = [r for r in ocr_results if r.confidence >= min_confidence]
        if not valid:
            return None
        valid.sort(key=lambda r: r.frame_index)
        text = " ".join(r.text for r in valid)
        return text[:max_length]

    @staticmethod
    def extract_transcript(transcript: Optional[str],
                           max_length: int = 10000) -> Optional[str]:
        if not transcript or not transcript.strip():
            return None
        return transcript[:max_length]
```

### 5.11 异常体系与错误处理（exceptions.py）

```python
class TextAnalysisError(Exception):
    """文本分析基础异常"""
    pass

class TextPreprocessingError(TextAnalysisError):
    """文本预处理失败"""
    pass

class SemanticEncodingError(TextAnalysisError):
    """语义编码失败"""
    pass

class ClassificationError(TextAnalysisError):
    """文本分类失败"""
    pass
```

### 5.12 核心分析器中的错误处理

`TextAnalyzer.analyze()` 使用 `asyncio.gather(return_exceptions=True)` 确保单个模块失败不阻断其他模块：

```python
class TextAnalyzer:
    """文本分析器 — 串联所有模块，并行执行互不依赖的分析"""

    def __init__(self, config: TextAnalysisConfig):
        self._preprocessor = TextPreprocessor()
        self._semantic = SemanticEncoder(config)
        self._sensitive = SensitiveMatcher(config)
        self._sentiment = SentimentAnalyzer(config)
        self._ner = NERecognizer(config)
        self._classifier = TextClassifier(config)

    async def analyze(self, text: str, source: str) -> SourceAnalysis:
        cleaned = self._preprocessor.process(text)

        # 并行执行5个模块，单个失败不影响其他
        results = await asyncio.gather(
            self._semantic.encode(cleaned.text),
            self._sensitive.match_all(cleaned.text),
            self._sentiment.analyze(cleaned.text),
            self._ner.recognize(cleaned.text),
            self._classifier.classify(cleaned.text),
            return_exceptions=True,
        )

        embed_result, sensitive_result, sentiment_result, ner_result, classify_result = results

        errors: List[str] = []

        # 逐个处理结果，异常则降级为 None/[] 并记录错误
        embedding = self._unwrap(embed_result, errors, "semantic_encoding")
        sensitive_words = self._unwrap(sensitive_result, errors, "sensitive_matching") or []
        sentiment_score = self._unwrap(sentiment_result, errors, "sentiment_analysis")
        entities = self._unwrap(ner_result, errors, "ner") or []
        category = self._unwrap(classify_result, errors, "classification")

        return SourceAnalysis(
            source=source,
            text_length=len(text),
            semantic_embedding=embedding,
            sensitive_words=sensitive_words,
            sentiment_score=sentiment_score,
            entities=entities,
            category=category,
            errors=errors,
        )

    def _unwrap(self, result: Any, errors: List[str], module: str) -> Any:
        """工具方法：异常→None + 记录错误"""
        if isinstance(result, Exception):
            errors.append(f"{module}: {result}")
            return None
        return result
```

**SourceAnalysis 新增 errors 字段**：

```python
class SourceAnalysis(BaseModel):
    """单来源文本分析结果"""
    source: str = Field(description="来源：ocr / transcript")
    text_length: int = Field(description="文本长度")
    semantic_embedding: Optional[List[float]] = Field(default=None, description="语义嵌入 (768维)")
    sensitive_words: List[SensitiveWord] = Field(default_factory=list, description="敏感词列表")
    sentiment_score: Optional[float] = Field(default=None, description="情感分数 -1~1")
    entities: List[Entity] = Field(default_factory=list, description="实体列表")
    category: Optional[CategoryResult] = Field(default=None, description="分类结果")
    errors: List[str] = Field(default_factory=list, description="各模块处理错误记录")
```

---

## 6. 降级逻辑

### 6.1 降级原则

| 原则 | 说明 |
|------|------|
| **模块级隔离** | 单个分析模块失败不影响其他模块 |
| **来源级隔离** | OCR 失败不影响语音分析，反之亦然 |
| **None 语义** | 所有失败场景统一返回 None（非空对象/空列表） |
| **错误可追溯** | 所有降级原因记录在 `SourceAnalysis.errors` 中 |

### 6.2 模块级降级矩阵

| 失败模块 | semantic_embedding | sensitive_words | sentiment_score | entities | category | 整体 |
|----------|:-:|:-:|:-:|:-:|:-:|------|
| 预处理 | None | [] | None | [] | None | **跳过该来源** |
| 语义编码 | None | ✅ | ✅ | ✅ | ✅ | 部分降级 |
| 敏感词匹配 | ✅ | [] | ✅ | ✅ | ✅ | 部分降级 |
| 情感分析 | ✅ | ✅ | None | ✅ | ✅ | 部分降级 |
| NER | ✅ | ✅ | ✅ | [] | ✅ | 部分降级 |
| 分类 | ✅ | ✅ | ✅ | ✅ | None | 部分降级 |

### 6.3 来源级降级矩阵

| 场景 | OCR 结果 | 语音结果 | violations | api 返回 |
|------|:--------:|:--------:|:----------:|:--------:|
| OCR + 语音都正常 | ✅ | ✅ | 合并两源 | 正常 |
| 只有 OCR 文本 | ✅ | None | 仅 OCR | `transcript=None` |
| 只有语音文本 | None | ✅ | 仅语音 | `ocr=None` |
| 都无文本 | None | None | [] | 空结果, `errors` 记录原因 |

### 6.4 实现机制

```python
# TextAnalyzer.analyze() — 模块级隔离
# 使用 asyncio.gather(return_exceptions=True) 捕获单个模块异常
results = await asyncio.gather(
    ...,
    return_exceptions=True,
)
# 每个结果解包：异常 → None + 记录 errors
```

### 6.5 构造时错误的处理

模型加载失败（如 BERT 下载失败、CUDA OOM）在 `TextAnalysisService.__init__()` 时直接抛出异常：

| 失败场景 | 处理方式 | 恢复手段 |
|----------|----------|----------|
| BERT 模型加载失败 | Service 构造抛 `OSError` | 检查网络/模型路径 |
| NER 模型加载失败 | Service 构造抛 `OSError` | 检查网络/模型路径 |
| 敏感词表文件缺失 | 记录 warning，使用空词表继续运行 | 补充词表文件后重启 |
| jieba 词典异常 | Service 构造抛 `ImportError` | 检查 jieba 安装 |

**设计意图**：构造时失败在启动期暴露，比运行时悄然降级更安全。

---

## 7. 并行执行与性能

### 7.1 执行模型

```
analyze_all() 调用
    │
    ├─ OCR 文本提取      (同步，轻量)
    ├─ 语音文本提取      (同步，轻量)
    │
    ├─ TextAnalyzer.analyze(ocr)
    │     │
    │     ├─ 预处理        (同步，轻量)
    │     ├─ 语义编码      (asyncio.to_thread)
    │     ├─ 敏感词匹配    (同步, O(n))
    │     ├─ 情感分析      (同步，轻量)
    │     ├─ NER          (asyncio.to_thread)
    │     └─ 分类          (asyncio.to_thread)
    │
    └─ TextAnalyzer.analyze(transcript)
          │
          └─ (同上)

    ResultMerger.merge()  (同步，轻量)
```

### 7.2 并行策略

| 层级 | 策略 | 说明 |
|------|------|------|
| 来源间 | OCR 和语音串行（避免 CPU 争抢） | 每个来源内部并行 |
| 模块间 | 5 个模块通过 `asyncio.gather` 并行 | 互不依赖的分析同时执行 |
| 模型推理 | `asyncio.to_thread` 放入线程池 | 不阻塞事件循环 |

**线程池资源估算**：
- 语义编码、NER、分类 各一个线程（共 3 个线程）
- 敏感词匹配和情感分析为同步快速路径，不占用线程池

---

## 8. 分类扩展机制

### 7.1 新增类别的步骤

1. **枚举**：`TextCategory` 添加新值
   ```python
   class TextCategory(str, Enum):
       PORN = "porn"
       VIOLENCE = "violence"
       POLITICAL = "political"
       AD = "ad"
       COPYRIGHT = "copyright"
       HARASSMENT = "harassment"  # 新增
       NORMAL = "normal"
   ```

2. **配置**：`TextAnalysisConfig.categories` 默认工厂自动跟随枚举

3. **模型**：使用新类别数据微调分类头（`num_labels` 自动 +1）

4. **无需修改的代码**：
   - `classifier.py` — `num_labels` 由配置动态决定
   - `CategoryResult` — `scores` 为 `Dict[str, float]`，自然包含新类别
   - `merger.py` — 无需改动

---

## 9. 模型与依赖

| 模块 | 依赖包 | 模型 | 模型大小 | 加载时机 |
|------|--------|------|----------|----------|
| 预处理 | `jieba` | 内置词典 | <10MB | Service 构造 |
| 语义编码 | `transformers`, `torch` | `bert-base-chinese` | ~400MB | Service 构造 |
| 敏感词匹配 | 内置 | 敏感词表文件 | 可配置 | Service 构造 |
| 情感分析 | `jieba` | 情感词典 | <1MB | Service 构造 |
| NER | `transformers`, `torch` | `ckiplab/bert-base-chinese-ner` | ~400MB | Service 构造 |
| 分类 | `transformers`, `torch` | 复用 BERT + 分类头 | ~400MB | Service 构造 |

**模型共享策略**：语义编码和分类器可共享同一个 BERT 编码器（`bert-base-chinese`），分类器只需额外加载一个分类头。实现时有两种选择：

| 方案 | 说明 | 内存 | 代码复杂度 |
| --- | --- | --- | --- |
| **独立加载（首版）** | 各自独立加载 | ~800MB | 低 |
| 共享权重 | 共用同一个 BERT 实例 | ~400MB | 中 |

**首版采用独立加载**，保证模块隔离性和代码清晰度；后续如遇内存瓶颈可合并。

---

## 10. 测试策略

### 10.1 单元测试

| 目标 | Mock 策略 | 关键用例 |
|------|-----------|----------|
| TextPreprocessor | 无 mock | HTML 清洗、分词、繁简转换、去停用词 |
| SemanticEncoder | mock transformers 模型 | 正常编码 / 空文本 / 超长截断 |
| SensitiveMatcher | 内置测试词表 | 精确命中 / 模糊命中 / 未命中 / 去重 |
| SentimentAnalyzer | 无 mock | 正面 / 负面 / 中性 / 否定词反转 |
| NERecognizer | mock transformers 模型 | 实体识别 / 无实体 / 多实体去重 |
| TextClassifier | mock transformers 模型 | 6 类分类 / 低置信度 / 空文本 |
| TextAnalyzer | mock 子模块 | 完整管道 / 空文本 / 错误传播 |
| TextAdaptor | 构造 OCRResult | 正常提取 / 低置信度过滤 / None |
| ResultMerger | mock SourceAnalysis | 两源合并 / 单源 / 无源 / 去重 |

### 10.2 测试专用策略

- 所有 transformers 模型通过 `unittest.mock.patch` 避免实际下载
- 敏感词表使用临时文件（`tmp_path` fixture）
- NER 和分类器的 mock 返回预定义的 logits 张量

---

## 11. 与周边模块的集成

### 11.1 上游集成

```python
# 预处理层输出 → 文本分析引擎
preprocessing_result = await preprocessing_service.process(input_result)

text_analysis_result = await text_analysis_service.analyze_all(
    video_id=preprocessing_result.input_id,
    ocr_results=preprocessing_result.ocr_results,
    transcript=preprocessing_result.transcript,
)
```

### 11.2 下游集成

`TextAnalysisResult` 将被规则引擎和 AI 引擎消费：
- **规则引擎**：使用 `violations` 和 `sensitive_words` 做规则匹配
- **AI 引擎**：使用 `semantic_embedding` 作为多模态特征的一部分

---

## 12. 自审清单

### 12.1 内部一致性
- ✅ `TextAnalysisResult` 字段与各模块输出一致
- ✅ `TextCategory` 枚举值与配置 `categories` 列表一致
- ✅ 异常体系与模块划分对应
- ✅ Service → TextAnalyzer → 各模块 调用链完整

### 12.2 范围检查
- ✅ 聚焦文本分析：不涉及图像、音频处理
- ✅ OCR 和语音作为输入来源，不负责提取
- ✅ 输出到 TextAnalysisResult，不跨层

### 12.3 模糊性检查
- ✅ OCR/语音 None 语义：来源缺失统一为跳过
- ✅ 模型加载时机：构造时预加载，文档明确标注
- ✅ 分类扩展路径：枚举 → 配置 → 微调，每步明确
