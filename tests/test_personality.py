"""Tests for the Freya personality module."""

import pytest

from freya.personality import (
    PersonalityBuilder,
    PersonalityMixer,
    TraitSystem,
    preset_creative,
    preset_friendly,
    preset_mentor,
    preset_professional,
)


class TestTraitSystem:
    """Tests for the TraitSystem."""

    def test_set_and_get_trait(self):
        ts = TraitSystem()
        ts.set_trait("humor", 0.8)
        assert ts.get_trait("humor") == 0.8

    def test_clamps_values(self):
        ts = TraitSystem()
        ts.set_trait("x", 1.5)
        assert ts.get_trait("x") == 1.0
        ts.set_trait("y", -0.3)
        assert ts.get_trait("y") == 0.0

    def test_remove_trait(self):
        ts = TraitSystem({"a": 0.5})
        assert ts.remove_trait("a") is True
        assert ts.has_trait("a") is False
        assert ts.remove_trait("a") is False

    def test_compatibility_identical(self):
        ts1 = TraitSystem({"a": 0.5, "b": 0.8})
        ts2 = TraitSystem({"a": 0.5, "b": 0.8})
        assert ts1.compatibility(ts2) == 1.0

    def test_compatibility_opposite(self):
        ts1 = TraitSystem({"a": 0.0})
        ts2 = TraitSystem({"a": 1.0})
        assert ts1.compatibility(ts2) == 0.0

    def test_compatibility_no_overlap(self):
        ts1 = TraitSystem({"a": 0.5})
        ts2 = TraitSystem({"b": 0.5})
        assert ts1.compatibility(ts2) == 0.5

    def test_to_dict(self):
        ts = TraitSystem({"humor": 0.7})
        d = ts.to_dict()
        assert d == {"humor": 0.7}


class TestPersonalityBuilder:
    """Tests for the PersonalityBuilder fluent API."""

    def test_basic_build(self):
        p = PersonalityBuilder().name("Test").tone("warm").build()
        assert p.name == "Test"
        assert p.tone == "warm"

    def test_fluent_chain(self):
        p = (
            PersonalityBuilder()
            .name("Chain")
            .friendliness(0.9)
            .formality(0.2)
            .humor(0.7)
            .empathy(0.8)
            .greeting("Hi!")
            .system_prompt("Be nice.")
            .build()
        )
        assert p.traits["friendliness"] == 0.9
        assert p.greeting == "Hi!"

    def test_trait_clamping(self):
        p = PersonalityBuilder().name("T").trait("x", 2.0).build()
        assert p.traits["x"] == 1.0


class TestPersonalityMixer:
    """Tests for the PersonalityMixer."""

    def test_equal_mix(self):
        a = PersonalityBuilder().name("A").friendliness(0.0).build()
        b = PersonalityBuilder().name("B").friendliness(1.0).build()
        mixed = PersonalityMixer.mix(a, b, ratio=0.5)
        assert abs(mixed.traits["friendliness"] - 0.5) < 0.01

    def test_mix_name(self):
        a = PersonalityBuilder().name("Alpha").build()
        b = PersonalityBuilder().name("Beta").build()
        mixed = PersonalityMixer.mix(a, b, name="Custom")
        assert mixed.name == "Custom"

    def test_mix_default_name(self):
        a = PersonalityBuilder().name("A").build()
        b = PersonalityBuilder().name("B").build()
        mixed = PersonalityMixer.mix(a, b)
        assert mixed.name == "A + B"

    def test_mix_invalid_ratio(self):
        a = PersonalityBuilder().name("A").build()
        b = PersonalityBuilder().name("B").build()
        with pytest.raises(ValueError, match="ratio"):
            PersonalityMixer.mix(a, b, ratio=1.5)

    def test_mix_skewed_toward_a(self):
        a = PersonalityBuilder().name("A").friendliness(1.0).build()
        b = PersonalityBuilder().name("B").friendliness(0.0).build()
        mixed = PersonalityMixer.mix(a, b, ratio=0.2)
        assert mixed.traits["friendliness"] > 0.7


class TestPresets:
    """Tests for personality presets."""

    def test_friendly_preset(self):
        p = preset_friendly()
        assert p.name == "Friendly"
        assert p.traits["friendliness"] >= 0.8

    def test_professional_preset(self):
        p = preset_professional()
        assert p.name == "Professional"
        assert p.traits["formality"] >= 0.8

    def test_creative_preset(self):
        p = preset_creative()
        assert p.name == "Creative"
        assert p.tone == "imaginative"

    def test_mentor_preset(self):
        p = preset_mentor()
        assert p.name == "Mentor"
        assert p.traits["empathy"] >= 0.8
