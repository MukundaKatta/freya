"""Tests for the Freya memory module."""

import pytest

from freya.memory import Fact, MemorySearch, MemoryStore


class TestFact:
    """Tests for the Fact dataclass."""

    def test_create_fact(self):
        f = Fact(content="The sky is blue", source="user", importance=0.7)
        assert f.content == "The sky is blue"
        assert f.source == "user"

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content must not be empty"):
            Fact(content="   ")

    def test_invalid_importance_raises(self):
        with pytest.raises(ValueError, match="Importance"):
            Fact(content="x", importance=1.5)

    def test_to_dict(self):
        f = Fact(content="test", tags=["a", "b"])
        d = f.to_dict()
        assert d["content"] == "test"
        assert d["tags"] == ["a", "b"]


class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_add_and_count(self):
        store = MemoryStore()
        store.add(Fact(content="fact one"))
        assert store.count == 1

    def test_capacity_eviction(self):
        store = MemoryStore(capacity=3)
        store.add(Fact(content="low importance", importance=0.1))
        store.add(Fact(content="medium importance", importance=0.5))
        store.add(Fact(content="high importance", importance=0.9))
        store.add(Fact(content="new fact", importance=0.6))
        assert store.count == 3
        contents = [f.content for f in store.facts]
        assert "low importance" not in contents

    def test_get_by_source(self):
        store = MemoryStore()
        store.add(Fact(content="from user", source="user"))
        store.add(Fact(content="from system", source="system"))
        results = store.get_by_source("user")
        assert len(results) == 1

    def test_get_by_importance(self):
        store = MemoryStore()
        store.add(Fact(content="low", importance=0.2))
        store.add(Fact(content="high", importance=0.9))
        results = store.get_by_importance(0.5)
        assert len(results) == 1
        assert results[0].content == "high"

    def test_get_by_tag(self):
        store = MemoryStore()
        store.add(Fact(content="tagged", tags=["python", "coding"]))
        store.add(Fact(content="untagged"))
        results = store.get_by_tag("python")
        assert len(results) == 1

    def test_clear(self):
        store = MemoryStore()
        store.add(Fact(content="x"))
        store.clear()
        assert store.count == 0


class TestMemorySearch:
    """Tests for MemorySearch."""

    def test_recall_by_keyword(self):
        store = MemoryStore()
        store.add(Fact(content="Python is a programming language", importance=0.8))
        store.add(Fact(content="Cats are cute animals", importance=0.5))
        search = MemorySearch(store)
        results = search.recall("Python programming")
        assert len(results) >= 1
        assert "Python" in results[0].content

    def test_recall_respects_min_importance(self):
        store = MemoryStore()
        store.add(Fact(content="low priority note", importance=0.1))
        store.add(Fact(content="high priority note", importance=0.9))
        search = MemorySearch(store)
        results = search.recall("note", min_importance=0.5)
        assert all(f.importance >= 0.5 for f in results)

    def test_recall_empty_query(self):
        store = MemoryStore()
        store.add(Fact(content="something"))
        search = MemorySearch(store)
        results = search.recall("")
        assert results == []

    def test_recall_by_tags(self):
        store = MemoryStore()
        store.add(Fact(content="tagged fact", tags=["work"], importance=0.8))
        store.add(Fact(content="other fact", tags=["personal"], importance=0.5))
        search = MemorySearch(store)
        results = search.recall_by_tags(["work"])
        assert len(results) == 1
        assert results[0].content == "tagged fact"

    def test_recall_max_results(self):
        store = MemoryStore()
        for i in range(10):
            store.add(Fact(content="fact about topic {}".format(i), importance=0.5))
        search = MemorySearch(store)
        results = search.recall("fact topic", max_results=3)
        assert len(results) <= 3
