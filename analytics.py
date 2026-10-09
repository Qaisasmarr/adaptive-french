"""Make recorded practice inspectable through per-word summaries and exports.

These statistics describe past attempts. They do not establish how well a
word will be remembered later or whether a scheduling policy is effective.

Public API: summarize returns one statistics row per vocabulary item;
export_stats writes those rows as CSV. Correctness is stored as 0 or 1, so
summing results counts correct answers and averaging times reports seconds.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import List, Sequence

from models import Review, Schedule


def summarize(schedule: Sequence[Schedule], reviews: Sequence[Review]) -> List[dict]:
    """Join current vocabulary to history, keeping unseen scores missing.

    A missing accuracy means 'not attempted', which differs from getting every
    answer wrong. Iterating the schedule also excludes removed vocabulary.
    """
    by_word = defaultdict(list)
    for review in reviews:
        by_word[review.word_id].append(review)
    rows = []
    for item in schedule:
        history = by_word[item.word.id]
        count = len(history)
        correct = sum(review.result for review in history)
        rows.append({
            "word_id": item.word.id,
            "french": item.word.french,
            "attempts": count,
            "correct": correct,
            "accuracy_pct": round(100 * correct / count, 1) if count else None,
            "avg_response_seconds": round(mean(r.response_time for r in history), 2) if count else None,
            "last_review_utc": max(r.timestamp for r in history).isoformat() if count else "",
            "next_review_utc": item.due_at.isoformat() if count else "",
            "interval_minutes": item.interval_minutes,
        })
    return rows


def export_stats(path: Path, rows: Sequence[dict]) -> None:
    """Write a spreadsheet-friendly snapshot; the CLI checks the output path.

    This helper itself opens in write mode, so direct callers must also guard
    against overwriting an existing file.
    """
    fields = ["word_id", "french", "attempts", "correct", "accuracy_pct",
              "avg_response_seconds", "last_review_utc", "next_review_utc", "interval_minutes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
