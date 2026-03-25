"""Long-term memory system for Freya AI companions.

Provides importance-scored fact storage and relevance-based recall
to give companions persistent knowledge across conversations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Fact:
    """A stored piece of knowledge with importance scoring.

    Attributes:
        content: The textual content of the fact.
        source: Where this fact originated (e.g. 'user', 'conversation').
        importance: Importance score from 0.0 (trivial) to 1.0 (critical).
        timestamp: Unix timestamp when the fact was recorded.
        tags: Optional tags for categorization.
    """

    content: str
    source: str = "unknown"
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate fact fields."""
        if not self.content or not self.content.strip():
            raise ValueError("Fact content must not be empty")
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(
                "Importance {} must be between 0.0 and 1.0".format(self.importance)
            )

    def age_seconds(self) -> float:
        """Return the age of this fact in seconds."""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict:
        """Serialize the fact to a dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "tags": list(self.tags),
        }


class MemoryStore:
    """Long-term memory storage with importance-based management.

    Stores facts with importance scores and provides retrieval by
    relevance, source, or recency.
    """

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize the memory store.

        Args:
            capacity: Maximum number of facts to retain.
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        self._facts: List[Fact] = []
        self._capacity = capacity

    @property
    def facts(self) -> List[Fact]:
        """Return a copy of all stored facts."""
        return list(self._facts)

    @property
    def count(self) -> int:
        """Return the number of stored facts."""
        return len(self._facts)

    def add(self, fact: Fact) -> None:
        """Add a fact, evicting the least important if at capacity."""
        if len(self._facts) >= self._capacity:
            self._evict_least_important()
        self._facts.append(fact)

    def _evict_least_important(self) -> None:
        """Remove the fact with the lowest importance score."""
        if not self._facts:
            return
        least = min(self._facts, key=lambda f: f.importance)
        self._facts.remove(least)

    def get_by_source(self, source: str) -> List[Fact]:
        """Return all facts from a given source."""
        return [f for f in self._facts if f.source == source]

    def get_by_importance(self, min_importance: float = 0.0) -> List[Fact]:
        """Return facts at or above the given importance threshold."""
        return [f for f in self._facts if f.importance >= min_importance]

    def get_by_tag(self, tag: str) -> List[Fact]:
        """Return all facts that have the given tag."""
        return [f for f in self._facts if tag in f.tags]

    def get_recent(self, count: int = 10) -> List[Fact]:
        """Return the most recently added facts."""
        return sorted(self._facts, key=lambda f: f.timestamp, reverse=True)[:count]

    def clear(self) -> None:
        """Remove all stored facts."""
        self._facts.clear()


class MemorySearch:
    """Search and recall facts from a MemoryStore by relevance.

    Uses simple keyword matching and importance weighting to rank
    facts by their relevance to a given query.
    """

    def __init__(self, store: MemoryStore) -> None:
        """Initialize with a reference to a MemoryStore."""
        self._store = store

    def recall(
        self,
        query: str,
        max_results: int = 5,
        min_importance: float = 0.0,
    ) -> List[Fact]:
        """Recall facts relevant to a query.

        Ranks facts by a combined score of keyword overlap and importance.

        Args:
            query: The search query string.
            max_results: Maximum number of facts to return.
            min_importance: Minimum importance threshold.

        Returns:
            List of facts sorted by relevance score (descending).
        """
        query_words = set(query.lower().split())
        if not query_words:
            return []

        scored: List[tuple] = []
        for fact in self._store.facts:
            if fact.importance < min_importance:
                continue
            fact_words = set(fact.content.lower().split())
            overlap = len(query_words & fact_words)
            if overlap == 0:
                continue
            keyword_score = overlap / max(len(query_words), 1)
            relevance = (keyword_score * 0.6) + (fact.importance * 0.4)
            scored.append((relevance, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:max_results]]

    def recall_by_tags(
        self, tags: List[str], max_results: int = 5
    ) -> List[Fact]:
        """Recall facts that match any of the given tags.

        Results are sorted by importance descending.
        """
        tag_set = set(tags)
        matching = [
            f for f in self._store.facts if tag_set & set(f.tags)
        ]
        matching.sort(key=lambda f: f.importance, reverse=True)
        return matching[:max_results]
