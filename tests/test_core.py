"""Tests for the Freya core module."""

import pytest

from freya.core import (
    Companion,
    CompanionSession,
    ConversationMemory,
    Message,
    Personality,
)


class TestPersonality:
    """Tests for the Personality dataclass."""

    def test_create_basic_personality(self):
        p = Personality(name="Test", traits={"friendliness": 0.8}, tone="warm")
        assert p.name == "Test"
        assert p.tone == "warm"
        assert p.traits["friendliness"] == 0.8

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name must not be empty"):
            Personality(name="  ")

    def test_invalid_trait_value_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Personality(name="Bad", traits={"x": 1.5})

    def test_get_trait_default(self):
        p = Personality(name="P")
        assert p.get_trait("missing", 0.7) == 0.7

    def test_set_trait(self):
        p = Personality(name="P")
        p.set_trait("humor", 0.9)
        assert p.traits["humor"] == 0.9

    def test_set_trait_invalid_raises(self):
        p = Personality(name="P")
        with pytest.raises(ValueError):
            p.set_trait("humor", -0.1)

    def test_describe(self):
        p = Personality(name="P", traits={"friendliness": 0.8, "formality": 0.2})
        desc = p.describe()
        assert "high friendliness" in desc
        assert "low formality" in desc

    def test_copy(self):
        p = Personality(name="P", traits={"a": 0.5})
        c = p.copy()
        c.traits["a"] = 0.9
        assert p.traits["a"] == 0.5


class TestMessage:
    """Tests for the Message dataclass."""

    def test_create_message(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="Role"):
            Message(role="invalid", content="x")

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content must not be empty"):
            Message(role="user", content="")

    def test_message_id_deterministic(self):
        m = Message(role="user", content="test", timestamp=1000.0)
        assert m.message_id == m.message_id

    def test_to_dict(self):
        m = Message(role="companion", content="Hi")
        d = m.to_dict()
        assert d["role"] == "companion"
        assert "message_id" in d


class TestConversationMemory:
    """Tests for ConversationMemory."""

    def test_add_and_count(self):
        mem = ConversationMemory()
        mem.add_message(Message(role="user", content="Hello"))
        assert mem.message_count == 1

    def test_get_recent(self):
        mem = ConversationMemory()
        for i in range(20):
            mem.add_message(Message(role="user", content="msg {}".format(i)))
        recent = mem.get_recent(5)
        assert len(recent) == 5

    def test_summarization_triggers(self):
        mem = ConversationMemory(max_messages=200, summary_threshold=10)
        for i in range(12):
            mem.add_message(Message(role="user", content="message about topic {}".format(i)))
        assert len(mem.summaries) >= 1

    def test_search(self):
        mem = ConversationMemory()
        mem.add_message(Message(role="user", content="I love Python"))
        mem.add_message(Message(role="user", content="Java is fine"))
        results = mem.search("python")
        assert len(results) == 1

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_message(Message(role="user", content="hello"))
        mem.clear()
        assert mem.message_count == 0

    def test_context_window(self):
        mem = ConversationMemory()
        mem.add_message(Message(role="user", content="short"))
        window = mem.get_context_window(2000)
        assert len(window) == 1


class TestCompanion:
    """Tests for the Companion engine."""

    def test_generate_response(self):
        p = Personality(name="Test", traits={"friendliness": 0.8})
        c = Companion(p)
        resp = c.generate_response("Tell me something")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_empty_input_raises(self):
        c = Companion(Personality(name="Test"))
        with pytest.raises(ValueError, match="must not be empty"):
            c.generate_response("")

    def test_response_count_increments(self):
        c = Companion(Personality(name="Test"))
        c.generate_response("Hello")
        c.generate_response("Again")
        assert c.response_count == 2


class TestCompanionSession:
    """Tests for CompanionSession."""

    def test_start_returns_greeting(self):
        p = Personality(name="Test", greeting="Hi there!")
        session = CompanionSession(Companion(p))
        greeting = session.start()
        assert greeting == "Hi there!"

    def test_send_message(self):
        p = Personality(name="Test", traits={"friendliness": 0.5})
        session = CompanionSession(Companion(p))
        session.start()
        resp = session.send_message("How are you?")
        assert isinstance(resp, str)

    def test_end_session(self):
        session = CompanionSession(Companion(Personality(name="Test")))
        session.start()
        summary = session.end()
        assert "session_id" in summary
        assert "message_count" in summary
        assert not session.is_active

    def test_send_after_end_raises(self):
        session = CompanionSession(Companion(Personality(name="Test")))
        session.start()
        session.end()
        with pytest.raises(RuntimeError, match="ended"):
            session.send_message("hello")

    def test_get_history(self):
        session = CompanionSession(Companion(Personality(name="Test")))
        session.start()
        session.send_message("test")
        history = session.get_history()
        assert len(history) >= 2
