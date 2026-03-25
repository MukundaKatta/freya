"""Freya - Self-hosted AI Companion Framework.

Norse Mythology: Freya, Goddess of Love and Magic.
Personality customization, conversation memory, and multi-modal interaction config.
"""

from freya.core import (
    Companion,
    CompanionSession,
    ConversationMemory,
    Message,
    Personality,
)
from freya.memory import Fact, MemorySearch, MemoryStore
from freya.personality import (
    PersonalityBuilder,
    PersonalityMixer,
    TraitSystem,
)

__version__ = "0.1.0"

__all__ = [
    "Companion",
    "CompanionSession",
    "ConversationMemory",
    "Fact",
    "Memory",
    "MemorySearch",
    "MemoryStore",
    "Message",
    "Personality",
    "PersonalityBuilder",
    "PersonalityMixer",
    "TraitSystem",
]
