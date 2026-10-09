"""Turn review history into an explainable rule-based practice schedule.

These rules work before there is enough history to train a recall model.
Keeping this module free of file I/O makes individual decisions easy to test.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Sequence

from models import Review, Schedule, Word

DAY = 24 * 60
MAX_INTERVAL = 30 * DAY


def next_interval(previous: float, review: Review, fast_seconds: float) -> float:
    """Return minutes until the next review, given the previous interval.

    Mistakes trigger a short retry delay. Correct answers extend the interval,
    with slower answers growing it more cautiously. These are tunable rules,
    not learned estimates of memory strength.
    """
    if review.result == 0:
        return 10.0
    if review.response_time > fast_seconds:
        return min(MAX_INTERVAL, max(DAY, previous * 1.5))
    # First fast success -> 1 day; second -> 3 days; then double.
    if previous < DAY:
        return float(DAY)
    if previous == DAY:
        return float(3 * DAY)
    return min(MAX_INTERVAL, previous * 2)


def build_schedule(words: Sequence[Word], reviews: Sequence[Review],
                   now: datetime, fast_seconds: float = 5.0) -> List[Schedule]:
    """Replay each word's history instead of maintaining a second state file.

    The review log stays the source of truth. Replaying also means changing
    fast_seconds applies the new threshold to past answers, not just new ones.
    """
    by_word = defaultdict(list)
    for review in sorted(reviews, key=lambda item: item.timestamp):
        by_word[review.word_id].append(review)
    schedule = []
    for word in words:
        history = by_word[word.id]
        if not history:
            schedule.append(Schedule(word, 0, 0.0, now, "new word"))
            continue
        interval = 0.0
        for review in history:
            interval = next_interval(interval, review, fast_seconds)
        latest = history[-1]
        if latest.result == 0:
            reason = "last answer incorrect"
        elif latest.response_time > fast_seconds:
            reason = "last answer correct but slow"
        else:
            reason = "last answer correct and fast"
        schedule.append(Schedule(word, len(history), interval,
                                 latest.timestamp + timedelta(minutes=interval), reason))
    return schedule


def select_due(schedule: Sequence[Schedule], now: datetime, limit: int) -> List[Schedule]:
    """Choose a bounded session, prioritizing existing review obligations."""
    # Overdue known words first, oldest due date first; new words afterwards.
    # Future reviews are deliberately excluded. Each word occurs once/session.
    due = [item for item in schedule if item.attempts == 0 or item.due_at <= now]
    due.sort(key=lambda item: (item.attempts == 0, item.due_at, item.word.id))
    return due[:limit]
