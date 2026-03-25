"""Core AI companion engine for Freya.

Provides the main companion session management, conversation memory,
personality-aware response generation, and message handling.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Personality:
    """Defines an AI companion's personality configuration.

    Attributes:
        name: Display name for the personality.
        traits: Dictionary of trait names to float values (0-1 scale).
        tone: Overall conversational tone descriptor.
        greeting: Default greeting message when a session begins.
        system_prompt: Base system prompt that shapes companion behavior.
    """

    name: str
    traits: Dict[str, float] = field(default_factory=dict)
    tone: str = "neutral"
    greeting: str = "Hello! How can I help you today?"
    system_prompt: str = "You are a helpful AI companion."

    def __post_init__(self) -> None:
        """Validate personality configuration after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Personality name must not be empty")
        for trait_name, value in self.traits.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    "Trait '{}' value {} must be between 0.0 and 1.0".format(
                        trait_name, value
                    )
                )

    def get_trait(self, trait_name: str, default: float = 0.5) -> float:
        """Return the value of a trait, or a default if not set."""
        return self.traits.get(trait_name, default)

    def set_trait(self, trait_name: str, value: float) -> None:
        """Set a trait value, enforcing the 0-1 range."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                "Trait value {} must be between 0.0 and 1.0".format(value)
            )
        self.traits[trait_name] = value

    def describe(self) -> str:
        """Return a human-readable description of this personality."""
        trait_parts = []
        for name, val in sorted(self.traits.items()):
            if val >= 0.7:
                trait_parts.append("high {}".format(name))
            elif val <= 0.3:
                trait_parts.append("low {}".format(name))
            else:
                trait_parts.append("moderate {}".format(name))
        trait_desc = ", ".join(trait_parts) if trait_parts else "no defined traits"
        return "{} (tone: {}, traits: {})".format(self.name, self.tone, trait_desc)

    def copy(self) -> "Personality":
        """Create a deep copy of this personality."""
        return Personality(
            name=self.name,
            traits=dict(self.traits),
            tone=self.tone,
            greeting=self.greeting,
            system_prompt=self.system_prompt,
        )


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The sender role ('user', 'companion', or 'system').
        content: The text content of the message.
        timestamp: Unix timestamp when the message was created.
        metadata: Optional additional data attached to the message.
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message fields."""
        valid_roles = ("user", "companion", "system")
        if self.role not in valid_roles:
            raise ValueError(
                "Role '{}' not in valid roles: {}".format(self.role, valid_roles)
            )
        if not self.content:
            raise ValueError("Message content must not be empty")

    @property
    def message_id(self) -> str:
        """Generate a deterministic ID based on content and timestamp."""
        raw = "{}:{}:{}".format(self.role, self.content, self.timestamp)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "message_id": self.message_id,
        }


class ConversationMemory:
    """Stores conversation history with automatic summarization.

    Manages a rolling window of messages and produces summaries
    when the history exceeds a configurable threshold.
    """

    def __init__(self, max_messages: int = 100, summary_threshold: int = 50) -> None:
        """Initialize conversation memory.

        Args:
            max_messages: Maximum messages to retain before pruning.
            summary_threshold: Number of messages that triggers summarization.
        """
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        if summary_threshold < 1:
            raise ValueError("summary_threshold must be at least 1")
        self._messages: List[Message] = []
        self._summaries: List[str] = []
        self._max_messages = max_messages
        self._summary_threshold = summary_threshold

    @property
    def messages(self) -> List[Message]:
        """Return a copy of the current message list."""
        return list(self._messages)

    @property
    def summaries(self) -> List[str]:
        """Return all generated summaries."""
        return list(self._summaries)

    @property
    def message_count(self) -> int:
        """Return the number of messages stored."""
        return len(self._messages)

    def add_message(self, message: Message) -> None:
        """Add a message and trigger summarization if needed."""
        self._messages.append(message)
        if len(self._messages) >= self._summary_threshold:
            self._summarize_and_prune()

    def _summarize_and_prune(self) -> None:
        """Summarize older messages and prune the history."""
        cutoff = len(self._messages) // 2
        old_messages = self._messages[:cutoff]
        topics = set()
        for msg in old_messages:
            words = msg.content.lower().split()
            for word in words:
                if len(word) > 4:
                    topics.add(word)
        topic_list = sorted(topics)[:5]
        summary = "Conversation covered {} messages. Topics: {}".format(
            len(old_messages),
            ", ".join(topic_list) if topic_list else "general discussion",
        )
        self._summaries.append(summary)
        self._messages = self._messages[cutoff:]

    def get_recent(self, count: int = 10) -> List[Message]:
        """Return the most recent messages up to count."""
        return self._messages[-count:]

    def get_context_window(self, max_tokens_estimate: int = 2000) -> List[Message]:
        """Return messages fitting within an estimated token budget.

        Uses a rough estimate of 4 characters per token.
        """
        result = []
        total = 0
        for msg in reversed(self._messages):
            estimated = len(msg.content) // 4
            if total + estimated > max_tokens_estimate:
                break
            result.append(msg)
            total += estimated
        result.reverse()
        return result

    def search(self, query: str) -> List[Message]:
        """Search messages for those containing the query string."""
        query_lower = query.lower()
        return [m for m in self._messages if query_lower in m.content.lower()]

    def clear(self) -> None:
        """Clear all messages and summaries."""
        self._messages.clear()
        self._summaries.clear()


class Companion:
    """AI companion engine that manages personality and generates responses.

    This is the core engine that combines a personality with conversation
    memory to produce contextually aware, personality-consistent responses.
    """

    def __init__(self, personality: Personality) -> None:
        """Initialize the companion with a personality."""
        self._personality = personality
        self._response_count = 0

    @property
    def personality(self) -> Personality:
        """Return the current personality."""
        return self._personality

    @property
    def response_count(self) -> int:
        """Return total responses generated."""
        return self._response_count

    def generate_response(
        self, user_input: str, context: Optional[List[Message]] = None
    ) -> str:
        """Generate a personality-aware response to user input.

        This is a simulated response generator that demonstrates how
        personality traits influence response style.

        Args:
            user_input: The user's message text.
            context: Optional conversation context messages.

        Returns:
            A generated response string.
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input must not be empty")

        self._response_count += 1
        friendliness = self._personality.get_trait("friendliness", 0.5)
        formality = self._personality.get_trait("formality", 0.5)
        humor = self._personality.get_trait("humor", 0.3)

        parts = []
        if friendliness >= 0.7:
            parts.append("I appreciate you sharing that!")
        elif friendliness <= 0.3:
            parts.append("Noted.")
        else:
            parts.append("Thank you for your message.")

        context_size = len(context) if context else 0
        if context_size > 0:
            parts.append(
                "Based on our conversation ({} messages),".format(context_size)
            )

        if formality >= 0.7:
            parts.append(
                "I would like to provide the following perspective"
                " on your inquiry regarding '{}'.".format(
                    user_input[:50]
                )
            )
        elif humor >= 0.6:
            parts.append(
                "Here's a thought about '{}' -- and yes,"
                " I promise it's a good one!".format(
                    user_input[:50]
                )
            )
        else:
            parts.append(
                "Regarding '{}', here's what I think.".format(user_input[:50])
            )

        return " ".join(parts)

    def get_greeting(self) -> str:
        """Return the personality's greeting message."""
        return self._personality.greeting


class CompanionSession:
    """Manages an ongoing interaction session with the companion.

    Combines a Companion engine with ConversationMemory to provide
    a stateful conversation experience.
    """

    def __init__(
        self,
        companion: Companion,
        memory: Optional[ConversationMemory] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize a companion session.

        Args:
            companion: The Companion engine to use.
            memory: Optional conversation memory (created if not provided).
            session_id: Optional unique session identifier.
        """
        self._companion = companion
        self._memory = memory or ConversationMemory()
        self._session_id = session_id or self._generate_session_id()
        self._active = True
        self._started_at = time.time()

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        raw = "session:{}".format(time.time())
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id

    @property
    def is_active(self) -> bool:
        """Return whether the session is still active."""
        return self._active

    @property
    def companion(self) -> Companion:
        """Return the companion engine."""
        return self._companion

    @property
    def memory(self) -> ConversationMemory:
        """Return the conversation memory."""
        return self._memory

    def start(self) -> str:
        """Start the session and return the greeting."""
        greeting = self._companion.get_greeting()
        greeting_msg = Message(role="companion", content=greeting)
        self._memory.add_message(greeting_msg)
        return greeting

    def send_message(self, user_input: str) -> str:
        """Send a user message and get a companion response.

        Args:
            user_input: The user's message text.

        Returns:
            The companion's response.

        Raises:
            RuntimeError: If the session has been ended.
        """
        if not self._active:
            raise RuntimeError("Session has been ended")

        user_msg = Message(role="user", content=user_input)
        self._memory.add_message(user_msg)

        context = self._memory.get_recent(10)
        response_text = self._companion.generate_response(user_input, context)

        response_msg = Message(role="companion", content=response_text)
        self._memory.add_message(response_msg)

        return response_text

    def end(self) -> Dict[str, Any]:
        """End the session and return a summary.

        Returns:
            Dictionary containing session statistics.
        """
        self._active = False
        duration = time.time() - self._started_at
        return {
            "session_id": self._session_id,
            "duration_seconds": round(duration, 2),
            "message_count": self._memory.message_count,
            "response_count": self._companion.response_count,
            "summaries": self._memory.summaries,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full conversation history as dictionaries."""
        return [m.to_dict() for m in self._memory.messages]
