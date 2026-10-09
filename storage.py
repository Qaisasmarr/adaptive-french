"""Keep vocabulary and raw attempts in local, inspectable CSV files.

Support the original headerless log so existing history remains usable.
Reject malformed records explicitly: silently dropping attempts would change
both the schedule and the statistics without the learner knowing.

Public API: load_words and load_reviews validate source records; is_correct
matches explicitly accepted translations; append_review saves one attempt.
Stable word IDs connect the vocabulary CSV to the append-only review history.
"""
import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from models import Review, Word

REVIEW_HEADER = ["word_id", "timestamp", "result", "response_time"]


class DataError(ValueError):
    """A file exists but contains an invalid record."""


def normalize(text: str) -> str:
    """Ignore case and spacing so formatting alone does not cost an answer."""
    return " ".join(text.strip().casefold().split())


def is_correct(answer: str, expected: str) -> bool:
    """Match only listed alternatives, keeping grading predictable."""
    # An explicit | separates accepted answers, e.g. "car|automobile".
    return normalize(answer) in {normalize(item) for item in expected.split("|")}


def load_words(path: Path) -> List[Word]:
    """Validate vocabulary IDs and answers before a quiz can record history."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"id", "french", "english"}.issubset(reader.fieldnames or []):
            raise DataError(f"{path}: expected headers id,french,english")
        words = []
        seen = set()
        for row in reader:
            values = [row.get(key) for key in ("id", "french", "english")]
            if not all(value and value.strip() for value in values):
                raise DataError(f"{path}: incomplete word at line {reader.line_num}")
            word = Word(*(value.strip() for value in values))
            if word.id in seen:
                raise DataError(f"{path}: duplicate word ID {word.id!r}")
            if any(not normalize(item) for item in word.english.split("|")):
                raise DataError(f"{path}: blank accepted answer for ID {word.id}")
            seen.add(word.id)
            words.append(word)
    return words


def parse_timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Original datetime.now() logs have no timezone. Interpret them as local
    # time on this computer; all new records include a UTC offset.
    return stamp.astimezone(timezone.utc)


def load_reviews(path: Path) -> List[Review]:
    """Read attempts in time order; a missing log means no practice yet."""
    if not path.exists():
        return []
    reviews = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        first_record = True
        for row in reader:
            if not row or all(not item.strip() for item in row):
                continue
            if first_record and row in (REVIEW_HEADER, ["id"] + REVIEW_HEADER[1:]):
                first_record = False
                continue
            first_record = False
            try:
                if len(row) != 4:
                    raise ValueError("expected four columns")
                word_id, timestamp, result, seconds = row
                result = int(result)
                seconds = float(seconds)
                if not word_id.strip() or result not in (0, 1):
                    raise ValueError("invalid ID or result; result must be 0 or 1")
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("response time must be a finite nonnegative number")
                reviews.append(Review(word_id.strip(), parse_timestamp(timestamp), result, seconds))
            except ValueError as error:
                raise DataError(f"{path}: invalid review at line {reader.line_num}: {error}") from error
    return sorted(reviews, key=lambda review: review.timestamp)


def append_review(path: Path, review: Review) -> None:
    """Append one answer without rewriting earlier attempts or their header.

    The quiz calls this after each answer so ending a session early does not
    discard answers already saved. This assumes one running writer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    # A valid hand-edited CSV need not end with a newline. Do not glue rows.
    needs_newline = False
    if not needs_header:
        with path.open("rb") as file:
            file.seek(-1, 2)
            needs_newline = file.read(1) not in (b"\n", b"\r")
    with path.open("a", encoding="utf-8", newline="") as file:
        if needs_newline:
            file.write("\n")
        writer = csv.writer(file)
        if needs_header:
            writer.writerow(REVIEW_HEADER)
        writer.writerow([review.word_id, review.timestamp.isoformat(timespec="seconds"),
                         review.result, round(review.response_time, 2)])
