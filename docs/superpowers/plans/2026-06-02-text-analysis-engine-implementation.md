# 文本分析引擎实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现文本分析引擎，对 OCR 文本和语音转写文本进行预处理、语义编码、敏感词检测、情感分析、实体识别和违规分类

**Architecture:** 模块化管道架构。TextAnalysisService 接收输入，通过 TextAnalyzer（内部串联 6 个分析模块，并行执行）分别处理 OCR 和语音来源，最后 ResultMerger 合并为 TextAnalysisResult。所有 transformers 模型构造时预加载，推理通过 asyncio.to_thread 放入线程池。

**Tech Stack:** Python 3.10+, asyncio, transformers, torch, jieba, opencc, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-06-02-text-analysis-engine-design.md`

---

## 文件结构

```
src/illegal_review/
├── data_models.py                                # [修改] TextAnalysisResult 更新 + 新增模型
└── config/settings.py                            # [修改] TextAnalysisConfig 更新
└── analysis_engine_layer/text_analysis/
    ├── __init__.py                               # [修改] 导出 TextAnalysisService
    ├── exceptions.py                             # [新建]
    ├── models.py                                 # [新建] PreprocessedText 内部模型
    ├── preprocessor.py                           # [新建] 文本预处理
    ├── semantic.py                               # [新建] BERT 语义编码
    ├── sensitive_matcher.py                      # [新建] AC 自动机敏感词匹配
    ├── sentiment.py                              # [新建] 情感分析（词典法）
    ├── ner.py                                    # [新建] 命名实体识别
    ├── classifier.py                             # [新建] 文本分类器
    ├── adaptor.py                                # [新建] 输入适配器
    ├── analyzer.py                               # [新建] TextAnalyzer 核心分析器
    ├── merger.py                                 # [新建] ResultMerger 结果合并
    └── service.py                                # [新建] TextAnalysisService

tests/analysis_engine_layer/text_analysis/
├── __init__.py                                   # [新建]
├── test_exceptions.py                            # [新建]
├── test_preprocessor.py                          # [新建]
├── test_semantic.py                              # [新建]
├── test_sensitive_matcher.py                     # [新建]
├── test_sentiment.py                             # [新建]
├── test_ner.py                                   # [新建]
├── test_classifier.py                            # [新建]
├── test_adaptor.py                               # [新建]
├── test_analyzer.py                              # [新建]
├── test_merger.py                                # [新建]
└── test_service.py                               # [新建]
```

---

### Task 1: 数据模型与类别枚举

**Files:**
- Modify: `src/illegal_review/data_models.py`
- Create: `tests/analysis_engine_layer/text_analysis/__init__.py`
- Test: 直接验证 (在 data_models.py 上修改)

- [ ] **Step 1: 新增文本分类枚举 TextCategory**

在 `data_models.py` 末尾（现有 `FeedbackData` 模型之后）添加：

```python
from enum import Enum


class TextCategory(str, Enum):
    """文本分类类别（str+Enum，可序列化为纯字符串）"""
    PORN = "porn"
    VIOLENCE = "violence"
    POLITICAL = "political"
    AD = "ad"
    COPYRIGHT = "copyright"
    NORMAL = "normal"
```

- [ ] **Step 2: 新增 SensitiveWord, Entity, CategoryResult 模型**

在 TextCategory 之后添加：

```python
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

- [ ] **Step 3: 新增 SourceAnalysis 模型**

在 CategoryResult 之后添加：

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

- [ ] **Step 4: 更新 TextAnalysisResult**

将现有的 `TextAnalysisResult` 替换为：

```python
class TextAnalysisResult(BaseModel):
    """文本分析结果"""
    video_id: UUID = Field(description="视频标识")
    ocr: Optional[SourceAnalysis] = Field(default=None, description="OCR文本分析结果")
    transcript: Optional[SourceAnalysis] = Field(default=None, description="语音转写分析结果")
    violations: List[ViolationDetection] = Field(default_factory=list, description="违规检测汇总")
    processing_stats: Dict[str, float] = Field(default_factory=dict, description="处理统计")
```

- [ ] **Step 5: 创建测试目录**

```bash
mkdir -p tests/analysis_engine_layer/text_analysis
touch tests/analysis_engine_layer/text_analysis/__init__.py
```

- [ ] **Step 6: 验证 data_models 导入正常**

```bash
python -c "from src.illegal_review.data_models import TextCategory, SensitiveWord, Entity, CategoryResult, SourceAnalysis, TextAnalysisResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/illegal_review/data_models.py tests/analysis_engine_layer/text_analysis/__init__.py
git commit -m "feat: add text analysis data models - TextCategory, SourceAnalysis, SensitiveWord, Entity, CategoryResult"
```

---

### Task 2: 配置文件

**Files:**
- Modify: `src/illegal_review/config/settings.py`
- Test: 直接验证

- [ ] **Step 1: 更新 TextAnalysisConfig**

将现有的 `TextAnalysisConfig` 替换为：

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
        default_factory=lambda: ["porn", "violence", "political", "ad", "copyright", "normal"]
    )
    classifier_threshold: float = 0.5

    # OCR 输入
    ocr_confidence_threshold: float = 0.5
    ocr_max_text_length: int = 5000

    # 批处理
    max_text_length: int = 10000
```

- [ ] **Step 2: 验证配置导入正常**

```bash
python -c "from src.illegal_review.config.settings import TextAnalysisConfig, SystemConfig; c = SystemConfig(); print(c.text_analysis.bert_model, c.text_analysis.categories)"
```

Expected: `bert-base-chinese ['porn', 'violence', 'political', 'ad', 'copyright', 'normal']`

- [ ] **Step 3: Commit**

```bash
git add src/illegal_review/config/settings.py
git commit -m "feat: update TextAnalysisConfig with full text analysis engine settings"
```

---

### Task 3: 异常体系

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/exceptions.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_exceptions.py`

- [ ] **Step 1: 编写异常测试**

```python
# tests/analysis_engine_layer/text_analysis/test_exceptions.py
import pytest
from src.illegal_review.analysis_engine_layer.text_analysis.exceptions import (
    TextAnalysisError,
    TextPreprocessingError,
    SemanticEncodingError,
    ClassificationError,
)


class TestTextAnalysisExceptions:
    def test_text_analysis_error_base(self):
        with pytest.raises(TextAnalysisError, match="base error"):
            raise TextAnalysisError("base error")

    def test_preprocessing_error_subclass(self):
        with pytest.raises(TextAnalysisError):
            raise TextPreprocessingError("preprocess failed")

    def test_semantic_encoding_error_subclass(self):
        with pytest.raises(TextAnalysisError):
            raise SemanticEncodingError("encoding failed")

    def test_classification_error_subclass(self):
        with pytest.raises(TextAnalysisError):
            raise ClassificationError("classify failed")

    def test_all_inherit_from_text_analysis_error(self):
        assert issubclass(TextPreprocessingError, TextAnalysisError)
        assert issubclass(SemanticEncodingError, TextAnalysisError)
        assert issubclass(ClassificationError, TextAnalysisError)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_exceptions.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现异常类**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/exceptions.py
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

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_exceptions.py -v
```

Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/exceptions.py tests/analysis_engine_layer/text_analysis/test_exceptions.py
git commit -m "feat: add text analysis exception hierarchy"
```

---

### Task 4: 内部数据模型

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/models.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_models.py`

- [ ] **Step 1: 编写内部模型测试**

```python
# tests/analysis_engine_layer/text_analysis/test_models.py
from src.illegal_review.analysis_engine_layer.text_analysis.models import PreprocessedText


class TestPreprocessedText:
    def test_create(self):
        pt = PreprocessedText(text="hello world", tokens=["hello", "world"])
        assert pt.text == "hello world"
        assert pt.tokens == ["hello", "world"]

    def test_empty(self):
        pt = PreprocessedText(text="", tokens=[])
        assert pt.text == ""
        assert pt.tokens == []
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_models.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 internal models**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/models.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessedText:
    """预处理后的文本（清洗+分词结果）"""
    text: str           # 清洗后的完整文本
    tokens: List[str]   # 分词结果
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_models.py -v
```

Expected: PASS (2/2)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/models.py tests/analysis_engine_layer/text_analysis/test_models.py
git commit -m "feat: add PreprocessedText internal model"
```

---

### Task 5: 文本预处理模块

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/preprocessor.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_preprocessor.py`

- [ ] **Step 1: 编写文本预处理测试**

```python
# tests/analysis_engine_layer/text_analysis/test_preprocessor.py
import pytest
from src.illegal_review.analysis_engine_layer.text_analysis.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return TextPreprocessor()

    def test_clean_html(self, preprocessor):
        result = preprocessor.clean("<p>hello</p>world")
        assert result == "hello world"

    def test_clean_special_chars(self, preprocessor):
        result = preprocessor.clean("hello!@#$world")
        assert result == "hello world"

    def test_clean_extra_spaces(self, preprocessor):
        result = preprocessor.clean("hello   world")
        assert result == "hello world"

    def test_segment(self, preprocessor):
        tokens = preprocessor.segment("我爱北京天安门")
        assert len(tokens) > 0
        assert "天安门" in tokens

    def test_normalize_traditional_to_simple(self, preprocessor):
        """繁体转简体"""
        result = preprocessor.normalize("簡體轉繁體")
        assert "简体" in result

    def test_normalize_fullwidth_to_halfwidth(self, preprocessor):
        """全角字母转半角"""
        result = preprocessor.normalize("ＨＥＬＬＯ")
        assert result == "HELLO"

    def test_process_returns_preprocessed_text(self, preprocessor):
        result = preprocessor.process("  Hello World!  ")
        assert result.text == "Hello World"
        assert len(result.tokens) > 0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_preprocessor.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 TextPreprocessor**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/preprocessor.py
import re
from typing import List, Optional, Set

import jieba

from src.illegal_review.analysis_engine_layer.text_analysis.models import PreprocessedText


class TextPreprocessor:
    """文本清洗与分词（使用 jieba，无额外模型依赖）"""

    _stopwords: Optional[Set[str]] = None

    def __init__(self, language: str = "zh"):
        self._jieba = jieba

    def clean(self, text: str) -> str:
        """去除HTML标签、特殊符号、多余空白"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s一-鿿]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def segment(self, text: str) -> List[str]:
        """分词"""
        return self._jieba.lcut(text)

    def _get_stopwords(self) -> Set[str]:
        """加载停用词表（类级别缓存）"""
        if TextPreprocessor._stopwords is None:
            # 内置基础停用词表
            TextPreprocessor._stopwords = {
                "的", "了", "在", "是", "我", "有", "和", "就",
                "不", "人", "都", "一", "一个", "上", "也", "很",
                "到", "说", "要", "去", "你", "会", "着", "没有",
                "看", "好", "自己", "这", "他", "她", "它", "们",
            }
        return TextPreprocessor._stopwords

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """去停用词"""
        stopwords = self._get_stopwords()
        return [t for t in tokens if t not in stopwords]

    def normalize(self, text: str) -> str:
        """繁简转换、全角半角统一"""
        import opencc
        converter = opencc.OpenCC('t2s.json')
        text = converter.convert(text)
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

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_preprocessor.py -v
```

Expected: PASS (7/7)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/preprocessor.py tests/analysis_engine_layer/text_analysis/test_preprocessor.py
git commit -m "feat: add TextPreprocessor with jieba segmentation and text cleaning"
```

---

### Task 6: AC自动机敏感词匹配

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/sensitive_matcher.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_sensitive_matcher.py`

- [ ] **Step 1: 编写敏感词匹配测试**

```python
# tests/analysis_engine_layer/text_analysis/test_sensitive_matcher.py
import pytest
import tempfile
from pathlib import Path
from src.illegal_review.analysis_engine_layer.text_analysis.sensitive_matcher import SensitiveMatcher
from src.illegal_review.config.settings import TextAnalysisConfig


@pytest.fixture
def word_list_path():
    """创建临时敏感词表"""
    content = [
        "毒品,illegal_drugs",
        "赌博,gambling",
        "暴力,violence",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(content))
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def config(word_list_path):
    return TextAnalysisConfig(
        sensitive_word_list_path=word_list_path,
        sensitive_fuzzy_match_enabled=False,
    )


class TestSensitiveMatcher:
    def test_match_exact(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match("这是一段包含毒品的文本")
        assert len(results) == 1
        assert results[0].word == "毒品"
        assert results[0].category == "illegal_drugs"

    def test_match_multiple(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match("毒品和赌博都是违法的")
        categories = {r.word for r in results}
        assert "毒品" in categories
        assert "赌博" in categories

    def test_no_match(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match("这是一段正常的文本")
        assert len(results) == 0

    def test_overlapping_match(self, config):
        """重叠敏感词应全部匹配"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("中国,location\n中国人,location")
            path = f.name

        cfg = TextAnalysisConfig(sensitive_word_list_path=path, sensitive_fuzzy_match_enabled=False)
        matcher = SensitiveMatcher(cfg)
        results = matcher.match("中国人")
        assert len(results) == 2
        Path(path).unlink(missing_ok=True)

    def test_match_all_without_fuzzy(self, config):
        matcher = SensitiveMatcher(config)
        results = matcher.match_all("毒品赌博")
        assert len(results) == 2

    def test_empty_word_list(self):
        """空词表不抛异常"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("")
            path = f.name

        cfg = TextAnalysisConfig(sensitive_word_list_path=path, sensitive_fuzzy_match_enabled=False)
        matcher = SensitiveMatcher(cfg)
        results = matcher.match("任何文本")
        assert len(results) == 0
        Path(path).unlink(missing_ok=True)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_sensitive_matcher.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 AC自动机敏感词匹配**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/sensitive_matcher.py
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.illegal_review.data_models import SensitiveWord
from src.illegal_review.config.settings import TextAnalysisConfig

logger = logging.getLogger(__name__)


class _ACTrieNode:
    """AC自动机节点"""
    __slots__ = ("children", "fail", "output")

    def __init__(self):
        self.children: Dict[str, "_ACTrieNode"] = {}
        self.fail: "_ACTrieNode" = None
        self.output: List[Tuple[str, str]] = []  # [(word, category), ...]


class SensitiveMatcher:
    """AC自动机敏感词匹配"""

    def __init__(self, config: TextAnalysisConfig):
        self._words: Dict[str, str] = {}
        self._fuzzy_enabled = config.sensitive_fuzzy_match_enabled
        self._fuzzy_threshold = config.sensitive_fuzzy_threshold
        self._root = _ACTrieNode()
        self._build_automaton(config.sensitive_word_list_path)

    def _build_automaton(self, path: str) -> None:
        """加载敏感词列表并构建 AC 自动机"""
        word_path = Path(path)
        if not word_path.exists():
            logger.warning(f"Sensitive word list not found: {path}, using empty list")
            return

        lines = word_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                word = parts[0].strip()
                category = parts[1].strip()
                self._add_word(word, category)

        self._build_fail_links()

    def _add_word(self, word: str, category: str) -> None:
        """向 Trie 中插入一个敏感词"""
        self._words[word] = category
        node = self._root
        for char in word:
            if char not in node.children:
                node.children[char] = _ACTrieNode()
            node = node.children[char]
        node.output.append((word, category))

    def _build_fail_links(self) -> None:
        """BFS 构建 fail 指针"""
        from collections import deque
        queue = deque()

        # 深度 1 的节点 fail 指向 root
        for child in self._root.children.values():
            child.fail = self._root
            queue.append(child)

        while queue:
            node = queue.popleft()
            for char, child in node.children.items():
                fail = node.fail
                while fail is not None and char not in fail.children:
                    fail = fail.fail
                child.fail = fail.children[char] if fail else self._root
                if child.fail:
                    child.output.extend(child.fail.output)
                queue.append(child)

    def match(self, text: str) -> List[SensitiveWord]:
        """AC自动机一次扫描，精确匹配"""
        results: List[SensitiveWord] = []
        node = self._root

        for i, char in enumerate(text):
            while node is not self._root and char not in node.children:
                node = node.fail
            if char in node.children:
                node = node.children[char]
            else:
                continue

            if node.output:
                for word, category in node.output:
                    start_pos = i - len(word) + 1
                    results.append(SensitiveWord(
                        word=word,
                        start_pos=start_pos,
                        end_pos=i + 1,
                        match_type="exact",
                        category=category,
                    ))

        return results

    def fuzzy_match(self, text: str) -> List[SensitiveWord]:
        """模糊匹配（基于编辑距离）"""
        if not self._fuzzy_enabled:
            return []

        results: List[SensitiveWord] = []
        for word, category in self._words.items():
            if len(word) < 2:
                continue
            # 滑动窗口检查文本片段与敏感词的编辑距离
            for i in range(len(text) - len(word) + 1):
                segment = text[i:i + len(word)]
                distance = self._levenshtein(segment, word)
                similarity = 1.0 - distance / max(len(word), 1)
                if similarity >= self._fuzzy_threshold:
                    results.append(SensitiveWord(
                        word=word,
                        start_pos=i,
                        end_pos=i + len(word),
                        match_type="fuzzy",
                        category=category,
                    ))
        return results

    def _levenshtein(self, s1: str, s2: str) -> int:
        """编辑距离计算"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def _deduplicate(self, results: List[SensitiveWord]) -> List[SensitiveWord]:
        """按置信度降序去重"""
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: len(x.word), reverse=True):
            key = (r.word, r.start_pos, r.end_pos)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    def match_all(self, text: str) -> List[SensitiveWord]:
        """精确 + 模糊匹配，结果合并去重"""
        results = self.match(text)
        if self._fuzzy_enabled:
            results.extend(self.fuzzy_match(text))
        return self._deduplicate(results)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_sensitive_matcher.py -v
```

Expected: PASS (7/7)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/sensitive_matcher.py tests/analysis_engine_layer/text_analysis/test_sensitive_matcher.py
git commit -m "feat: add SensitiveMatcher with AC automaton exact and fuzzy matching"
```

---

### Task 7: 情感分析模块

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/sentiment.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_sentiment.py`

- [ ] **Step 1: 编写情感分析测试**

```python
# tests/analysis_engine_layer/text_analysis/test_sentiment.py
import pytest
from src.illegal_review.analysis_engine_layer.text_analysis.sentiment import SentimentAnalyzer
from src.illegal_review.config.settings import TextAnalysisConfig


class TestSentimentAnalyzer:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig()

    @pytest.fixture
    def analyzer(self, config):
        return SentimentAnalyzer(config)

    def test_positive_text(self, analyzer):
        score = analyzer.analyze("今天天气真好，心情非常愉快")
        assert score > 0

    def test_negative_text(self, analyzer):
        score = analyzer.analyze("太糟糕了，真是令人愤怒")
        assert score < 0

    def test_neutral_text(self, analyzer):
        score = analyzer.analyze("今天星期二")
        assert -0.3 <= score <= 0.3

    def test_negation_reversal(self, analyzer):
        """否定词反转情感极性"""
        negative = analyzer.analyze("不好")
        positive = analyzer.analyze("好")
        assert negative < positive

    def test_adverb_weight(self, analyzer):
        """程度副词增强情感强度"""
        strong = analyzer.analyze("非常喜欢")
        weak = analyzer.analyze("喜欢")
        assert strong > weak

    def test_empty_text(self, analyzer):
        score = analyzer.analyze("")
        assert score == 0.0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_sentiment.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 SentimentAnalyzer**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/sentiment.py
import re
from typing import Dict, Set

import jieba
from src.illegal_review.config.settings import TextAnalysisConfig


class SentimentAnalyzer:
    """情感分析 — 基于 lexicon + 规则（无模型依赖）"""

    def __init__(self, config: TextAnalysisConfig):
        self._positive_words: Set[str] = {
            "好", "喜欢", "爱", "优秀", "美丽", "漂亮", "棒", "赞",
            "开心", "高兴", "快乐", "幸福", "美好", "精彩", "完美",
            "出色", "享受", "舒适", "满意", "欣赏", "喜爱",
        }
        self._negative_words: Set[str] = {
            "坏", "差", "讨厌", "恨", "垃圾", "恶心", "丑陋",
            "伤心", "难过", "痛苦", "愤怒", "生气", "糟糕",
            "恐怖", "可怕", "严重", "危险", "失败", "崩溃",
            "不良", "违法", "违规", "非法",
        }
        self._negation_words: Set[str] = {
            "不", "没", "别", "勿", "毋", "未", "无", "不是",
            "没有", "不要", "不会", "不能", "不该",
        }
        self._adverb_boost: Dict[str, float] = {
            "非常": 1.5, "很": 1.3, "太": 1.4, "极其": 1.8,
            "特别": 1.5, "十分": 1.4, "相当": 1.3, "无比": 1.8,
            "有点": 0.7, "稍微": 0.6, "比较": 0.8, "不太": 0.5,
        }

    def analyze(self, text: str) -> float:
        """返回 -1.0 ~ 1.0，正值=正面，负值=负面，0=中性"""
        if not text or not text.strip():
            return 0.0

        words = jieba.lcut(text)
        score = 0.0
        negation_distance = 0  # 否定词影响范围（向后几个词）

        for i, w in enumerate(words):
            # 处理否定词
            if w in self._negation_words:
                negation_distance = 2
                continue

            # 处理程度副词
            weight = 1.0
            if w in self._adverb_boost:
                weight = self._adverb_boost[w]
                continue

            # 计算情感得分
            if w in self._positive_words:
                if negation_distance > 0:
                    score -= weight * 0.5  # 否定+正面=负面
                else:
                    score += weight * 1.0
            elif w in self._negative_words:
                if negation_distance > 0:
                    score += weight * 0.5  # 否定+负面=正面
                else:
                    score -= weight * 1.0

            if negation_distance > 0:
                negation_distance -= 1

        # 归一化到 [-1, 1]
        return max(-1.0, min(1.0, score / max(len(words), 1) * 5))
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_sentiment.py -v
```

Expected: PASS (6/6)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/sentiment.py tests/analysis_engine_layer/text_analysis/test_sentiment.py
git commit -m "feat: add SentimentAnalyzer with lexicon-based sentiment scoring"
```

---

### Task 8: BERT 语义编码模块

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/semantic.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_semantic.py`

- [ ] **Step 1: 编写语义编码测试（mock transformers）**

```python
# tests/analysis_engine_layer/text_analysis/test_semantic.py
import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.semantic import SemanticEncoder
from src.illegal_review.config.settings import TextAnalysisConfig


class TestSemanticEncoder:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(bert_model="bert-base-chinese")

    @pytest.fixture
    def mock_model(self):
        """Mock transformers 模型返回固定 768 维向量"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        # 模拟 [CLS] 向量: 768 维
        import torch
        mock_output.last_hidden_state[:, 0, :].squeeze.return_value = torch.zeros(768)
        mock_model.return_value = mock_output

        return mock_tokenizer, mock_model

    @pytest.mark.asyncio
    async def test_encode_returns_768d_vector(self, config, mock_model):
        mock_tokenizer, mock_model = mock_model
        with patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoModel.from_pretrained", return_value=mock_model):
            encoder = SemanticEncoder(config)
            embedding = await encoder.encode("测试文本")
            assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_encode_empty_text(self, config, mock_model):
        mock_tokenizer, mock_model = mock_model
        with patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.semantic.AutoModel.from_pretrained", return_value=mock_model):
            encoder = SemanticEncoder(config)
            embedding = await encoder.encode("")
            assert len(embedding) == 768
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_semantic.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 SemanticEncoder**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/semantic.py
import asyncio
import logging
from typing import List

from transformers import AutoTokenizer, AutoModel

from src.illegal_review.config.settings import TextAnalysisConfig

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """BERT 语义编码器 — 文本 → 768维语义向量"""

    def __init__(self, config: TextAnalysisConfig):
        logger.info(f"Loading BERT model: {config.bert_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
        self._model = AutoModel.from_pretrained(config.bert_model)
        self._model.eval()
        logger.info(f"BERT model '{config.bert_model}' loaded")

    async def encode(self, text: str) -> List[float]:
        """文本 → 768维语义向量（线程池推理）"""
        return await asyncio.to_thread(self._encode_sync, text)

    def _encode_sync(self, text: str) -> List[float]:
        import torch
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        if isinstance(embedding, float):
            embedding = [embedding]
        return embedding
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_semantic.py -v
```

Expected: PASS (2/2)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/semantic.py tests/analysis_engine_layer/text_analysis/test_semantic.py
git commit -m "feat: add SemanticEncoder with BERT [CLS] embedding"
```

---

### Task 9: 实体识别模块

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/ner.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_ner.py`

- [ ] **Step 1: 编写 NER 测试（mock transformers）**

```python
# tests/analysis_engine_layer/text_analysis/test_ner.py
import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.ner import NERecognizer
from src.illegal_review.config.settings import TextAnalysisConfig


class TestNERecognizer:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(ner_enabled=True)

    @pytest.mark.asyncio
    async def test_recognize_returns_entities(self, config):
        """Mock 模型返回预定义的 BIO 标签"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "张", "三", "去", "北", "京", "[SEP]"]

        mock_model = MagicMock()
        import torch
        # 模拟 logits: (1, 7, num_labels)
        mock_model.config.id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
        mock_logits = torch.zeros(1, 7, 5)
        mock_logits[0, 1, 1] = 10  # "张" → B-PER
        mock_logits[0, 2, 2] = 10  # "三" → I-PER
        mock_logits[0, 4, 3] = 10  # "北" → B-LOC
        mock_logits[0, 5, 4] = 10  # "京" → I-LOC
        mock_model.return_value.logits = mock_logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoModelForTokenClassification.from_pretrained", return_value=mock_model):
            recognizer = NERecognizer(config)
            entities = await recognizer.recognize("张三去北京")

            assert len(entities) == 2
            names = {e.name for e in entities}
            assert "张三" in names
            assert "北京" in names

    @pytest.mark.asyncio
    async def test_no_entities(self, config):
        """无实体时返回空列表"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "正", "常", "文", "本", "[SEP]"]

        mock_model = MagicMock()
        import torch
        mock_model.config.id2label = {0: "O"}
        mock_logits = torch.zeros(1, 6, 1)
        mock_model.return_value.logits = mock_logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.ner.AutoModelForTokenClassification.from_pretrained", return_value=mock_model):
            recognizer = NERecognizer(config)
            entities = await recognizer.recognize("正常文本")
            assert len(entities) == 0
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_ner.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 NERecognizer**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/ner.py
import asyncio
import logging
from typing import Dict, List, Optional

from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.illegal_review.data_models import Entity
from src.illegal_review.config.settings import TextAnalysisConfig

logger = logging.getLogger(__name__)

# NER 标签前缀 → 实体类型映射
_TAG_TO_TYPE = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "TIME": "time",
}


class NERecognizer:
    """命名实体识别 — 使用 ckiplab/bert-base-chinese-ner"""

    def __init__(self, config: TextAnalysisConfig):
        if not config.ner_enabled:
            logger.info("NER disabled by config")
            self._model = None
            self._tokenizer = None
            return

        logger.info("Loading NER model: ckiplab/bert-base-chinese-ner")
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self._model = AutoModelForTokenClassification.from_pretrained(
            "ckiplab/bert-base-chinese-ner"
        )
        self._model.eval()
        self._id2label: Dict[int, str] = self._model.config.id2label
        logger.info("NER model loaded")

    async def recognize(self, text: str) -> List[Entity]:
        """识别命名实体（线程池推理）"""
        if self._model is None:
            return []
        return await asyncio.to_thread(self._recognize_sync, text)

    def _recognize_sync(self, text: str) -> List[Entity]:
        import torch
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]

        tokens = self._tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist()
        )

        # 解码 BIO 标签
        entities: List[Entity] = []
        current_entity: Optional[dict] = None

        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            label = self._id2label.get(pred, "O")

            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue

            # 特殊 token 过滤
            if token.startswith("##"):
                if current_entity:
                    current_entity["name"] += token[2:]
                continue

            if label.startswith("B-"):
                # 保存前一个实体
                if current_entity:
                    entities.append(self._make_entity(current_entity, text))
                entity_type = _TAG_TO_TYPE.get(label[2:], "other")
                current_entity = {"name": token, "type": entity_type, "start_pos": None}

            elif label.startswith("I-") and current_entity:
                current_entity["name"] += token

            else:
                if current_entity:
                    entities.append(self._make_entity(current_entity, text))
                    current_entity = None

        if current_entity:
            entities.append(self._make_entity(current_entity, text))

        return entities

    def _make_entity(self, entity_info: dict, text: str) -> Entity:
        """构造 Entity，在原文中查找位置"""
        name = entity_info["name"]
        start_pos = text.find(name)
        if start_pos == -1:
            start_pos = 0
        return Entity(
            name=name,
            type=entity_info["type"],
            start_pos=start_pos,
            end_pos=start_pos + len(name),
            confidence=1.0,
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_ner.py -v
```

Expected: PASS (2/2)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/ner.py tests/analysis_engine_layer/text_analysis/test_ner.py
git commit -m "feat: add NERecognizer with ckiplab BERT NER model"
```

---

### Task 10: 文本分类器

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/classifier.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_classifier.py`

- [ ] **Step 1: 编写文本分类测试（mock transformers）**

```python
# tests/analysis_engine_layer/text_analysis/test_classifier.py
import pytest
from unittest.mock import patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.classifier import TextClassifier
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.data_models import TextCategory


class TestTextClassifier:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(
            categories=["porn", "violence", "political", "ad", "copyright", "normal"],
            classifier_threshold=0.5,
        )

    @pytest.mark.asyncio
    async def test_classify_returns_category(self, config):
        """Mock 模型返回 normal 类别"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        import torch
        # 模拟 logits：normal 类最高
        logits = torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0, 5.0]])
        mock_model.return_value.logits = logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            classifier = TextClassifier(config)
            result = await classifier.classify("正常文本")
            assert result.category == TextCategory.NORMAL
            assert result.confidence > 0.5
            assert len(result.scores) == 6

    @pytest.mark.asyncio
    async def test_classify_porn(self, config):
        """分类为 porn"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        import torch
        logits = torch.tensor([[5.0, -2.0, -2.0, -2.0, -2.0, -2.0]])
        mock_model.return_value.logits = logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            classifier = TextClassifier(config)
            result = await classifier.classify("色情内容")
            assert result.category == TextCategory.PORN

    @pytest.mark.asyncio
    async def test_classify_empty_text(self, config):
        """空文本返回 normal"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        import torch
        logits = torch.tensor([[-2.0, -2.0, -2.0, -2.0, -2.0, 5.0]])
        mock_model.return_value.logits = logits

        with patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.classifier.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model):
            classifier = TextClassifier(config)
            result = await classifier.classify("")
            assert result.category == TextCategory.NORMAL
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_classifier.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 TextClassifier**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/classifier.py
import asyncio
import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.illegal_review.data_models import CategoryResult
from src.illegal_review.config.settings import TextAnalysisConfig

logger = logging.getLogger(__name__)


class TextClassifier:
    """文本违规分类器 — bert + 分类头"""

    def __init__(self, config: TextAnalysisConfig):
        self._categories = config.categories
        self._threshold = config.classifier_threshold
        model_name = config.classifier_model_path or config.bert_model

        logger.info(f"Loading text classifier: {model_name} (num_labels={len(self._categories)})")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self._categories)
        )
        self._model.eval()
        logger.info("Text classifier loaded")

    async def classify(self, text: str) -> CategoryResult:
        """文本分类 → CategoryResult（线程池推理）"""
        return await asyncio.to_thread(self._classify_sync, text)

    def _classify_sync(self, text: str) -> CategoryResult:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        scores = {cat: float(probs[i]) for i, cat in enumerate(self._categories)}
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        return CategoryResult(
            category=self._categories[pred_idx],
            confidence=confidence,
            scores=scores,
        )
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_classifier.py -v
```

Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/classifier.py tests/analysis_engine_layer/text_analysis/test_classifier.py
git commit -m "feat: add TextClassifier with BERT sequence classification"
```

---

### Task 11: 输入适配器

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/adaptor.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_adaptor.py`

- [ ] **Step 1: 编写适配器测试**

```python
# tests/analysis_engine_layer/text_analysis/test_adaptor.py
import pytest
from uuid import uuid4
from src.illegal_review.analysis_engine_layer.text_analysis.adaptor import TextAdaptor
from src.illegal_review.data_models import OCRResult


class TestTextAdaptor:
    def test_extract_ocr_text_merges_multiple(self):
        """多个 OCRResult 按帧索引排序合并"""
        results = [
            OCRResult(text="世界", confidence=0.9, bbox=None, frame_index=5),
            OCRResult(text="你好", confidence=0.95, bbox=None, frame_index=3),
        ]
        text = TextAdaptor.extract_ocr_text(results)
        assert text == "你好 世界"

    def test_extract_ocr_text_filters_low_confidence(self):
        """低于阈值的结果被过滤"""
        results = [
            OCRResult(text="正常", confidence=0.9, bbox=None, frame_index=0),
            OCRResult(text="噪声", confidence=0.3, bbox=None, frame_index=1),
        ]
        text = TextAdaptor.extract_ocr_text(results, min_confidence=0.5)
        assert text == "正常"

    def test_extract_ocr_text_empty(self):
        """空列表返回 None"""
        assert TextAdaptor.extract_ocr_text([]) is None

    def test_extract_ocr_text_all_filtered(self):
        """全部过滤后返回 None"""
        results = [
            OCRResult(text="噪声", confidence=0.2, bbox=None, frame_index=0),
        ]
        assert TextAdaptor.extract_ocr_text(results, min_confidence=0.5) is None

    def test_extract_ocr_text_max_length(self):
        """超长截断"""
        results = [
            OCRResult(text="a" * 100, confidence=0.9, bbox=None, frame_index=0),
        ]
        text = TextAdaptor.extract_ocr_text(results, max_length=10)
        assert len(text) == 10

    def test_extract_transcript_normal(self):
        text = TextAdaptor.extract_transcript("Hello World")
        assert text == "Hello World"

    def test_extract_transcript_empty(self):
        assert TextAdaptor.extract_transcript("") is None
        assert TextAdaptor.extract_transcript(None) is None
        assert TextAdaptor.extract_transcript("  ") is None

    def test_extract_transcript_max_length(self):
        text = TextAdaptor.extract_transcript("a" * 200, max_length=50)
        assert len(text) == 50
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_adaptor.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 TextAdaptor**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/adaptor.py
from typing import List, Optional

from src.illegal_review.data_models import OCRResult


class TextAdaptor:
    """将 PreprocessingResult 中的文本提取为标准字符串"""

    @staticmethod
    def extract_ocr_text(
        ocr_results: Optional[List[OCRResult]],
        min_confidence: float = 0.5,
        max_length: int = 5000,
    ) -> Optional[str]:
        """从 OCR 结果提取文本：排序 → 过滤置信度 → 合并 → 截断"""
        if not ocr_results:
            return None
        valid = [r for r in ocr_results if r.confidence >= min_confidence]
        if not valid:
            return None
        valid.sort(key=lambda r: r.frame_index)
        text = " ".join(r.text for r in valid)
        return text[:max_length]

    @staticmethod
    def extract_transcript(
        transcript: Optional[str],
        max_length: int = 10000,
    ) -> Optional[str]:
        """提取转写文本，仅做空值和长度处理"""
        if not transcript or not transcript.strip():
            return None
        return transcript[:max_length]
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_adaptor.py -v
```

Expected: PASS (8/8)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/adaptor.py tests/analysis_engine_layer/text_analysis/test_adaptor.py
git commit -m "feat: add TextAdaptor for OCR and transcript text extraction"
```

---

### Task 12: 结果合并器

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/merger.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_merger.py`

- [ ] **Step 1: 编写合并器测试**

```python
# tests/analysis_engine_layer/text_analysis/test_merger.py
import pytest
from uuid import uuid4
from src.illegal_review.analysis_engine_layer.text_analysis.merger import ResultMerger
from src.illegal_review.data_models import (
    SourceAnalysis, SensitiveWord, CategoryResult, ViolationDetection,
    TextCategory,
)


class TestResultMerger:
    @pytest.fixture
    def merger(self):
        return ResultMerger()

    @pytest.fixture
    def video_id(self):
        return uuid4()

    def test_merge_both_sources(self, merger, video_id):
        ocr = SourceAnalysis(
            source="ocr", text_length=10,
            sensitive_words=[
                SensitiveWord(word="广告", start_pos=0, end_pos=2, match_type="exact", category="ad"),
            ],
            category=CategoryResult(category=TextCategory.AD, confidence=0.9, scores={"ad": 0.9}),
        )
        transcript = SourceAnalysis(
            source="transcript", text_length=20,
            sensitive_words=[],
            category=CategoryResult(category=TextCategory.NORMAL, confidence=0.95, scores={"normal": 0.95}),
        )

        result = merger.merge(ocr, transcript, video_id)
        assert result.ocr is not None
        assert result.transcript is not None
        # OCR 的广告敏感词 + 分类 → 2 violations
        assert len(result.violations) >= 1

    def test_merge_ocr_only(self, merger, video_id):
        ocr = SourceAnalysis(source="ocr", text_length=5)
        result = merger.merge(ocr, None, video_id)
        assert result.ocr is not None
        assert result.transcript is None

    def test_merge_transcript_only(self, merger, video_id):
        transcript = SourceAnalysis(source="transcript", text_length=5)
        result = merger.merge(None, transcript, video_id)
        assert result.ocr is None
        assert result.transcript is not None

    def test_merge_both_none(self, merger, video_id):
        result = merger.merge(None, None, video_id)
        assert result.ocr is None
        assert result.transcript is None
        assert result.violations == []

    def test_deduplicate_same_category(self, merger, video_id):
        """相同 type+category 取最高置信度"""
        ocr = SourceAnalysis(
            source="ocr", text_length=5,
            sensitive_words=[
                SensitiveWord(word="毒品", start_pos=0, end_pos=2, match_type="exact", category="drug"),
                SensitiveWord(word="毒品", start_pos=0, end_pos=2, match_type="exact", category="drug"),
            ],
        )
        result = merger.merge(ocr, None, video_id)
        # 两个相同敏感词去重后剩 1 条
        drug_violations = [v for v in result.violations if v.category == "drug"]
        assert len(drug_violations) <= 1
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_merger.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 ResultMerger**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/merger.py
import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from src.illegal_review.data_models import (
    SourceAnalysis, TextAnalysisResult, ViolationDetection,
    TextCategory,
)

logger = logging.getLogger(__name__)


class ResultMerger:
    """合并 OCR 和语音的分析结果"""

    def merge(
        self,
        ocr_result: Optional[SourceAnalysis],
        transcript_result: Optional[SourceAnalysis],
        video_id: UUID,
    ) -> TextAnalysisResult:
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

        violations = self._deduplicate(violations)

        return TextAnalysisResult(
            video_id=video_id,
            ocr=ocr_result,
            transcript=transcript_result,
            violations=violations,
            processing_stats={
                "violations_count": float(len(violations)),
                "has_ocr": float(ocr_result is not None),
                "has_transcript": float(transcript_result is not None),
            },
        )

    def _deduplicate(self, violations: List[ViolationDetection]) -> List[ViolationDetection]:
        """相同 (type, category) 取最高置信度"""
        seen: Dict[Tuple[str, str], ViolationDetection] = {}
        for v in violations:
            key = (v.type, v.category)
            if key not in seen or v.confidence > seen[key].confidence:
                seen[key] = v
        return list(seen.values())
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_merger.py -v
```

Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/merger.py tests/analysis_engine_layer/text_analysis/test_merger.py
git commit -m "feat: add ResultMerger with source merging and violation dedup"
```

---

### Task 13: 核心分析器

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/analyzer.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_analyzer.py`

- [ ] **Step 1: 编写 TextAnalyzer 测试**

```python
# tests/analysis_engine_layer/text_analysis/test_analyzer.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.analyzer import TextAnalyzer
from src.illegal_review.config.settings import TextAnalysisConfig


class TestTextAnalyzer:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(
            sensitive_fuzzy_match_enabled=False,
            sensitive_word_list_path="",
        )

    @pytest.fixture
    def analyzer(self, config):
        with patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.SemanticEncoder"), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.SensitiveMatcher"), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.analyzer.TextClassifier"):
            return TextAnalyzer(config)

    @pytest.mark.asyncio
    async def test_analyze_returns_source_analysis(self, analyzer):
        result = await analyzer.analyze("测试文本", source="ocr")
        assert result.source == "ocr"
        assert result.text_length == 4

    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, analyzer):
        result = await analyzer.analyze("", source="ocr")
        assert result is not None
        assert result.text_length == 0

    @pytest.mark.asyncio
    async def test_analyze_multiple_sources(self, analyzer):
        ocr_result = await analyzer.analyze("OCR文本", source="ocr")
        transcript_result = await analyzer.analyze("语音文本", source="transcript")
        assert ocr_result.source == "ocr"
        assert transcript_result.source == "transcript"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_analyzer.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 TextAnalyzer**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/analyzer.py
import asyncio
import logging
from typing import Any, List

from src.illegal_review.data_models import SourceAnalysis
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.analysis_engine_layer.text_analysis.preprocessor import TextPreprocessor
from src.illegal_review.analysis_engine_layer.text_analysis.semantic import SemanticEncoder
from src.illegal_review.analysis_engine_layer.text_analysis.sensitive_matcher import SensitiveMatcher
from src.illegal_review.analysis_engine_layer.text_analysis.sentiment import SentimentAnalyzer
from src.illegal_review.analysis_engine_layer.text_analysis.ner import NERecognizer
from src.illegal_review.analysis_engine_layer.text_analysis.classifier import TextClassifier

logger = logging.getLogger(__name__)


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
        """完整分析流水线：预处理 → 5模块并行 → SourceAnalysis"""
        cleaned = self._preprocessor.process(text)

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
        """异常 → None + 记录错误"""
        if isinstance(result, Exception):
            errors.append(f"{module}: {result}")
            return None
        return result
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_analyzer.py -v
```

Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/analyzer.py tests/analysis_engine_layer/text_analysis/test_analyzer.py
git commit -m "feat: add TextAnalyzer with parallel module orchestration and degradation"
```

---

### Task 14: Service 层与模块导出

**Files:**
- Create: `src/illegal_review/analysis_engine_layer/text_analysis/service.py`
- Modify: `src/illegal_review/analysis_engine_layer/text_analysis/__init__.py`
- Create: `tests/analysis_engine_layer/text_analysis/test_service.py`

- [ ] **Step 1: 编写 Service 测试**

```python
# tests/analysis_engine_layer/text_analysis/test_service.py
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock
from src.illegal_review.analysis_engine_layer.text_analysis.service import TextAnalysisService
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.data_models import OCRResult, TextAnalysisResult


class TestTextAnalysisService:
    @pytest.fixture
    def config(self):
        return TextAnalysisConfig(
            sensitive_fuzzy_match_enabled=False,
            sensitive_word_list_path="",
        )

    @pytest.fixture
    def service(self, config):
        with patch("src.illegal_review.analysis_engine_layer.text_analysis.service.TextAnalyzer"), \
             patch("src.illegal_review.analysis_engine_layer.text_analysis.service.ResultMerger"):
            return TextAnalysisService(config)

    @pytest.mark.asyncio
    async def test_analyze_all_with_ocr_and_transcript(self, service):
        """OCR + 语音都有的正常流程"""
        video_id = uuid4()
        ocr_results = [
            OCRResult(text="测试文字", confidence=0.9, bbox=None, frame_index=0),
        ]
        transcript = "这是一段语音转写文本"

        # Mock analyzer.analyze 返回简单的 SourceAnalysis
        from src.illegal_review.data_models import SourceAnalysis
        mock_result = SourceAnalysis(source="ocr", text_length=4)
        service._analyzer.analyze = AsyncMock(return_value=mock_result)

        # Mock merger.merge 返回 TextAnalysisResult
        service._merger.merge = MagicMock(return_value=TextAnalysisResult(
            video_id=video_id,
            ocr=mock_result,
        ))

        result = await service.analyze_all(video_id, ocr_results, transcript)
        assert isinstance(result, TextAnalysisResult)
        assert result.video_id == video_id

    @pytest.mark.asyncio
    async def test_analyze_all_no_inputs(self, service):
        """无 OCR 无语音 → 空结果"""
        video_id = uuid4()
        service._merger.merge = MagicMock(return_value=TextAnalysisResult(
            video_id=video_id,
        ))
        result = await service.analyze_all(video_id, ocr_results=None, transcript=None)
        assert result.video_id == video_id
        assert result.ocr is None
        assert result.transcript is None
```

- [ ] **Step 2: 运行测试验证失败**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_service.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 TextAnalysisService**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/service.py
import logging
from typing import List, Optional
from uuid import UUID

from src.illegal_review.data_models import (
    OCRResult, TextAnalysisResult,
)
from src.illegal_review.config.settings import TextAnalysisConfig
from src.illegal_review.analysis_engine_layer.text_analysis.analyzer import TextAnalyzer
from src.illegal_review.analysis_engine_layer.text_analysis.adaptor import TextAdaptor
from src.illegal_review.analysis_engine_layer.text_analysis.merger import ResultMerger

logger = logging.getLogger(__name__)


class TextAnalysisService:
    """文本分析引擎对外门面"""

    def __init__(self, config: TextAnalysisConfig):
        self._config = config
        self._analyzer = TextAnalyzer(config)
        self._merger = ResultMerger()

    async def analyze_all(
        self,
        video_id: UUID,
        ocr_results: Optional[List[OCRResult]] = None,
        transcript: Optional[str] = None,
    ) -> TextAnalysisResult:
        """完整分析：OCR + 语音，自动跳过 None 来源"""
        tasks = {}

        if ocr_results:
            ocr_text = TextAdaptor.extract_ocr_text(
                ocr_results,
                min_confidence=self._config.ocr_confidence_threshold,
                max_length=self._config.ocr_max_text_length,
            )
            if ocr_text:
                tasks["ocr"] = self._analyzer.analyze(ocr_text, source="ocr")

        if transcript:
            trans_text = TextAdaptor.extract_transcript(
                transcript,
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
            video_id=video_id,
        )
```

- [ ] **Step 3b: 更新 `__init__.py`**

```python
# src/illegal_review/analysis_engine_layer/text_analysis/__init__.py
"""
文本分析引擎模块

负责文本内容分析：
- BERT语义分析：理解文本语义
- 敏感词匹配：AC自动机检测敏感词汇
- 情感分析：基于词典分析文本情感倾向
- 实体识别：命名实体识别（人物/地点/组织/时间）
- 文本分类：6类违规文本分类（可扩展）

上游：预处理层（OCR文本 + 语音转写文本）
下游：规则引擎、AI引擎
"""

from src.illegal_review.analysis_engine_layer.text_analysis.service import TextAnalysisService

__all__ = ["TextAnalysisService"]
```

- [ ] **Step 4: 运行测试验证通过**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/test_service.py -v
```

Expected: PASS (2/2)

- [ ] **Step 5: Commit**

```bash
git add src/illegal_review/analysis_engine_layer/text_analysis/service.py src/illegal_review/analysis_engine_layer/text_analysis/__init__.py tests/analysis_engine_layer/text_analysis/test_service.py
git commit -m "feat: add TextAnalysisService with OCR/transcript pipeline orchestration"
```

---

### Task 15: 集成验证

- [ ] **Step 1: 运行全部测试**

```bash
python -m pytest tests/analysis_engine_layer/text_analysis/ -v
```

Expected: ALL PASS（总计约 55+ 个测试用例）

- [ ] **Step 2: 验证模块导入**

```bash
python -c "from src.illegal_review.analysis_engine_layer.text_analysis import TextAnalysisService; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: 运行全项目测试确保无回归**

```bash
python -m pytest tests/ -v
```

Expected: 所有 130 个已有测试继续通过，文本分析引擎测试全部通过

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "feat: complete text analysis engine implementation with 6 analysis modules"
```

---

## 依赖关系图

```
Task 1 (data_models)  ──→ Task 2 (config)
  │                         │
  └─────────┬───────────────┘
            │
            ├──→ Task 3 (exceptions)
            ├──→ Task 4 (internal models)
            │
            ├──→ Task 5 (preprocessor)
            ├──→ Task 6 (sensitive_matcher)
            ├──→ Task 7 (sentiment)
            ├──→ Task 8 (semantic)
            ├──→ Task 9 (ner)
            └──→ Task 10 (classifier)
                      │
                      ├──→ Task 11 (adaptor)
                      ├──→ Task 12 (merger)
                      ├──→ Task 13 (analyzer)
                      └──→ Task 14 (service)
                                │
                                └──→ Task 15 (integration)
```

**并行机会**：Tasks 1-4 为基础层（需顺序）。Tasks 5-10 分析模块之间互不依赖，可并行实现。Tasks 11-14 依赖 Tasks 5-10。
