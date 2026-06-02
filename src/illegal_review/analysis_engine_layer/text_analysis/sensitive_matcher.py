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
