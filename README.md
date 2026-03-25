# Freya -- AI Companion Framework

> Norse Mythology: Freya, Goddess of Love and Magic | Self-hosted AI companion with personality customization

[![GitHub Pages](https://img.shields.io/badge/Live_Demo-Visit_Site-blue?style=for-the-badge)](https://MukundaKatta.github.io/freya/)
[![GitHub](https://img.shields.io/github/license/MukundaKatta/freya?style=flat-square)](LICENSE)

## Overview

Freya is a self-hosted AI companion framework that provides personality customization, conversation memory, and multi-modal interaction configuration. Build AI companions with distinct personalities, long-term memory, and contextually aware responses.

## Features

- **Personality System** -- Fluent builder API for creating custom personalities with traits on a 0-1 scale
- **Conversation Memory** -- Rolling conversation history with automatic summarization
- **Long-term Memory** -- Importance-scored fact storage with relevance-based recall
- **Personality Mixing** -- Blend two personalities with configurable ratios
- **Built-in Presets** -- Ready-to-use personalities: friendly, professional, creative, mentor

## Quick Start

```python
from freya import Companion, CompanionSession
from freya.personality import PersonalityBuilder

# Build a custom personality
personality = (
    PersonalityBuilder()
    .name("Nova")
    .tone("warm")
    .friendliness(0.9)
    .humor(0.7)
    .empathy(0.8)
    .greeting("Hey! What shall we explore today?")
    .build()
)

# Start a session
session = CompanionSession(Companion(personality))
greeting = session.start()
response = session.send_message("Tell me something interesting")
```

## Installation

```bash
git clone https://github.com/MukundaKatta/freya.git
cd freya
PYTHONPATH=src python3 -c "import freya; print(freya.__version__)"
```

## Running Tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Project Structure

```
src/freya/
    __init__.py          # Package exports
    core.py              # Companion engine, session, conversation memory
    personality.py       # Personality builder, traits, mixer, presets
    memory.py            # Long-term memory store and search
tests/
    test_core.py         # Core module tests
    test_personality.py  # Personality system tests
    test_memory.py       # Memory module tests
```

## Live Demo

Visit the landing page: **https://MukundaKatta.github.io/freya/**

## License

MIT License -- Officethree Technologies

## Part of the Mythological Portfolio

This is project **#freya** in the [100-project Mythological Portfolio](https://github.com/MukundaKatta) by Officethree Technologies.
