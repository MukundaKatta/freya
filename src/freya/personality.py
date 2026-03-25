"""Personality system for Freya AI companions.

Provides fluent builder API, trait management, personality mixing,
and built-in presets for common companion archetypes.
"""

from __future__ import annotations

from typing import Dict, Optional

from freya.core import Personality


# Default trait ranges used by the trait system
_DEFAULT_TRAITS = {
    "friendliness": 0.5,
    "formality": 0.5,
    "humor": 0.3,
    "empathy": 0.5,
}


class TraitSystem:
    """Manages personality traits on a 0-1 continuous scale.

    Each trait represents a behavioral dimension. The system provides
    validation, normalization, and compatibility checking between
    trait configurations.
    """

    def __init__(self, traits: Optional[Dict[str, float]] = None) -> None:
        """Initialize the trait system with optional starting traits."""
        self._traits: Dict[str, float] = {}
        if traits:
            for name, value in traits.items():
                self.set_trait(name, value)

    @property
    def traits(self) -> Dict[str, float]:
        """Return a copy of the current trait map."""
        return dict(self._traits)

    def set_trait(self, name: str, value: float) -> None:
        """Set a trait, clamping the value to [0, 1]."""
        clamped = max(0.0, min(1.0, value))
        self._traits[name] = round(clamped, 4)

    def get_trait(self, name: str, default: float = 0.5) -> float:
        """Get a trait value, returning a default if not set."""
        return self._traits.get(name, default)

    def remove_trait(self, name: str) -> bool:
        """Remove a trait. Returns True if it existed."""
        if name in self._traits:
            del self._traits[name]
            return True
        return False

    def has_trait(self, name: str) -> bool:
        """Check whether a trait is defined."""
        return name in self._traits

    def compatibility(self, other: "TraitSystem") -> float:
        """Compute compatibility score (0-1) with another trait system.

        Uses the average of 1 minus the absolute difference for each
        shared trait. Traits present in only one system are ignored.
        """
        shared = set(self._traits.keys()) & set(other._traits.keys())
        if not shared:
            return 0.5  # neutral if no overlap
        total = sum(
            1.0 - abs(self._traits[t] - other._traits[t]) for t in shared
        )
        return round(total / len(shared), 4)

    def to_dict(self) -> Dict[str, float]:
        """Serialize traits to a dictionary."""
        return dict(self._traits)


class PersonalityBuilder:
    """Fluent API for constructing Personality instances.

    Allows chained method calls for a clean, readable construction pattern.
    """

    def __init__(self) -> None:
        """Initialize the builder with defaults."""
        self._name: str = "Companion"
        self._traits: Dict[str, float] = dict(_DEFAULT_TRAITS)
        self._tone: str = "neutral"
        self._greeting: str = "Hello! How can I help you today?"
        self._system_prompt: str = "You are a helpful AI companion."

    def name(self, name: str) -> "PersonalityBuilder":
        """Set the personality name."""
        self._name = name
        return self

    def tone(self, tone: str) -> "PersonalityBuilder":
        """Set the conversational tone."""
        self._tone = tone
        return self

    def greeting(self, greeting: str) -> "PersonalityBuilder":
        """Set the greeting message."""
        self._greeting = greeting
        return self

    def system_prompt(self, prompt: str) -> "PersonalityBuilder":
        """Set the system prompt."""
        self._system_prompt = prompt
        return self

    def trait(self, name: str, value: float) -> "PersonalityBuilder":
        """Set a specific trait value."""
        self._traits[name] = max(0.0, min(1.0, value))
        return self

    def friendliness(self, value: float) -> "PersonalityBuilder":
        """Set the friendliness trait."""
        return self.trait("friendliness", value)

    def formality(self, value: float) -> "PersonalityBuilder":
        """Set the formality trait."""
        return self.trait("formality", value)

    def humor(self, value: float) -> "PersonalityBuilder":
        """Set the humor trait."""
        return self.trait("humor", value)

    def empathy(self, value: float) -> "PersonalityBuilder":
        """Set the empathy trait."""
        return self.trait("empathy", value)

    def build(self) -> Personality:
        """Construct and return the Personality instance."""
        return Personality(
            name=self._name,
            traits=dict(self._traits),
            tone=self._tone,
            greeting=self._greeting,
            system_prompt=self._system_prompt,
        )


class PersonalityMixer:
    """Blend two personalities together with a configurable ratio.

    Produces a new personality whose traits are weighted averages
    of the source personalities.
    """

    @staticmethod
    def mix(
        a: Personality,
        b: Personality,
        ratio: float = 0.5,
        name: Optional[str] = None,
    ) -> Personality:
        """Blend two personalities.

        Args:
            a: First source personality.
            b: Second source personality.
            ratio: Blend ratio (0.0 = all A, 1.0 = all B).
            name: Optional name for the result. Defaults to 'A + B'.

        Returns:
            A new blended Personality.
        """
        if not (0.0 <= ratio <= 1.0):
            raise ValueError("Mix ratio must be between 0.0 and 1.0")

        all_trait_names = set(a.traits.keys()) | set(b.traits.keys())
        blended_traits: Dict[str, float] = {}
        for trait_name in all_trait_names:
            val_a = a.get_trait(trait_name, 0.5)
            val_b = b.get_trait(trait_name, 0.5)
            blended_traits[trait_name] = round(
                val_a * (1.0 - ratio) + val_b * ratio, 4
            )

        result_name = name or "{} + {}".format(a.name, b.name)
        tone = a.tone if ratio < 0.5 else b.tone
        greeting = a.greeting if ratio <= 0.5 else b.greeting

        return Personality(
            name=result_name,
            traits=blended_traits,
            tone=tone,
            greeting=greeting,
            system_prompt=a.system_prompt if ratio < 0.5 else b.system_prompt,
        )


def preset_friendly() -> Personality:
    """Return a friendly companion personality preset."""
    return (
        PersonalityBuilder()
        .name("Friendly")
        .tone("warm")
        .greeting("Hey there! Great to see you! What's on your mind?")
        .friendliness(0.9)
        .formality(0.2)
        .humor(0.7)
        .empathy(0.8)
        .system_prompt("You are a warm, friendly AI companion who is enthusiastic and supportive.")
        .build()
    )


def preset_professional() -> Personality:
    """Return a professional companion personality preset."""
    return (
        PersonalityBuilder()
        .name("Professional")
        .tone("formal")
        .greeting("Good day. How may I assist you?")
        .friendliness(0.5)
        .formality(0.9)
        .humor(0.1)
        .empathy(0.4)
        .system_prompt("You are a professional AI assistant focused on clear and precise communication.")
        .build()
    )


def preset_creative() -> Personality:
    """Return a creative companion personality preset."""
    return (
        PersonalityBuilder()
        .name("Creative")
        .tone("imaginative")
        .greeting("Welcome to the realm of ideas! What shall we explore?")
        .friendliness(0.7)
        .formality(0.3)
        .humor(0.6)
        .empathy(0.6)
        .system_prompt("You are a creative AI companion who thinks outside the box and inspires new ideas.")
        .build()
    )


def preset_mentor() -> Personality:
    """Return a mentor companion personality preset."""
    return (
        PersonalityBuilder()
        .name("Mentor")
        .tone("encouraging")
        .greeting("Hello! I'm here to help you learn and grow. What would you like to work on?")
        .friendliness(0.8)
        .formality(0.5)
        .humor(0.3)
        .empathy(0.9)
        .system_prompt("You are a patient AI mentor who guides learning through questions and encouragement.")
        .build()
    )
