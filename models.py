"""Shared records keep file handling, scheduling, and reporting independent.

Frozen dataclasses prevent accidental field changes as records move between
modules. A Schedule is derived from reviews; it is not stored as source data.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Word:
    """Vocabulary with a stable ID linking it to reviews, even if rows move."""

    id: str
    french: str
    english: str


@dataclass(frozen=True)
class Review:
    """One answered prompt: result is 0 or 1; response_time is in seconds."""

    word_id: str
    timestamp: datetime
    result: int
    response_time: float


@dataclass(frozen=True)
class Schedule:
    """A word's computed review state, including a reason for display."""

    word: Word
    attempts: int
    interval_minutes: float
    due_at: datetime
    reason: str
