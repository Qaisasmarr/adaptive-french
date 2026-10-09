"""Command-line entry point: quiz, schedule, stats, and train.

This module connects the pieces and handles user interaction. Storage,
scheduling, summaries, and model fitting live separately so they can be used
and tested without an interactive quiz.
"""
import argparse
import csv
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from analytics import export_stats, summarize
from models import Review
from recall_model import load_model, probabilities, train
from scheduler import build_schedule, select_due
from storage import DataError, append_review, is_correct, load_reviews, load_words

DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return number


def run_quiz(items, review_path: Path, strategy: str) -> None:
    """Ask a preselected queue and save only submitted answers as attempts.

    Keeping selection outside this loop lets all strategies share the same
    timing, grading, and logging behavior.
    """
    print(f"\nAdaptive French | {strategy} practice | {len(items)} words")
    print("Type :q to finish or :skip to skip. Each answered word is saved immediately.")
    results = []
    for index, item in enumerate(items, start=1):
        print(f"\n[{index}/{len(items)}] French: {item.word.french}")
        # A monotonic timer measures duration even if the wall clock changes.
        started = time.perf_counter()
        try:
            answer = input("English: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        seconds = time.perf_counter() - started
        if answer.casefold() == ":q":
            break
        if answer.casefold() == ":skip":
            print("Skipped; no attempt recorded.")
            continue
        correct = int(is_correct(answer, item.word.english))
        review = Review(item.word.id, datetime.now(timezone.utc), correct, seconds)
        append_review(review_path, review)
        results.append(review)
        if correct:
            print(f"Correct! ({seconds:.2f}s)")
        else:
            print(f"Incorrect. Accepted answer: {item.word.english.replace('|', ' / ')} ({seconds:.2f}s)")
    print("\n--- Quiz Results ---")
    count = len(results)
    correct = sum(review.result for review in results)
    print(f"Correct answers: {correct}/{count}")
    if count:
        print(f"Accuracy: {100 * correct / count:.1f}%")
        print(f"Average response time: {mean(r.response_time for r in results):.2f}s")
    else:
        print("No answers recorded.")


def show_stats(schedule, reviews) -> list:
    """Display progress for current vocabulary and return rows for export."""
    rows = summarize(schedule, reviews)
    known_ids = {item.word.id for item in schedule}
    current = [review for review in reviews if review.word_id in known_ids]
    answered = [row for row in rows if row["attempts"]]
    print(f"\nVocabulary: {len(rows)} words | Practiced: {len(answered)} | Attempts: {len(current)}")
    if current:
        print(f"Overall accuracy: {100 * sum(r.result for r in current) / len(current):.1f}%")
        print(f"Average response time: {mean(r.response_time for r in current):.2f}s")
        print("\nWords with lowest historical accuracy (small samples may be misleading):")
        weakest = sorted(answered, key=lambda row: (row["accuracy_pct"], -row["attempts"], row["french"]))
        for row in weakest[:5]:
            print(f"  {row['french']}: {row['accuracy_pct']:.1f}% over {row['attempts']} attempts; "
                  f"{row['avg_response_seconds']:.2f}s average")
    else:
        print("No history yet. Start with: python3 quiz.py quiz")
    return rows


def main(argv=None) -> int:
    """Validate options, load shared data, and dispatch the requested command."""
    parser = argparse.ArgumentParser(description="Adaptive French: CSV-based vocabulary practice and analytics.")
    parser.add_argument("command", nargs="?", choices=["quiz", "schedule", "stats", "train"], default="quiz")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="folder containing words.csv and reviews.csv")
    parser.add_argument("--limit", type=positive_int, default=10, help="maximum words per session")
    parser.add_argument("--strategy", choices=["adaptive", "random", "predictive"], default="adaptive",
                        help="adaptive chooses due words; predictive ranks due words by estimated recall; random samples all words")
    parser.add_argument("--model", type=Path, help="model JSON to read; for train, a new output path")
    parser.add_argument("--fast-seconds", type=positive_float, default=5.0, help="heuristic fast-answer threshold")
    parser.add_argument("--seed", type=int, help="reproducible random word selection")
    parser.add_argument("--export", type=Path, help="write stats to a new CSV path (stats command only)")
    args = parser.parse_args(argv)
    if args.export and args.command != "stats":
        parser.error("--export is only available for stats")
    if args.strategy == "predictive" and not args.model:
        parser.error("predictive practice requires --model pointing to a trained JSON file")
    if args.command == "train" and not args.model:
        parser.error("train requires --model pointing to a new output JSON path")
    if args.command == "train" and args.model.exists():
        parser.error("model output must be a new file; choose a new name for each training run")
    words_path = args.data_dir / "words.csv"
    reviews_path = args.data_dir / "reviews.csv"
    if args.export:
        # Stats are derived data. Never overwrite the learner's source history.
        if args.export.resolve() in (words_path.resolve(), reviews_path.resolve()) or args.export.exists():
            parser.error("export path must be a new file, separate from words.csv and reviews.csv")
    try:
        words = load_words(words_path)
        reviews = load_reviews(reviews_path)
        known_ids = {word.id for word in words}
        orphans = sum(review.word_id not in known_ids for review in reviews)
        if orphans:
            print(f"Note: {orphans} reviews reference absent word IDs; they are preserved but excluded from stats.")
        now = datetime.now(timezone.utc)
        if any(review.timestamp > now for review in reviews):
            print("Note: some reviews are dated in the future. Check your timestamps or timezone.")
        schedule = build_schedule(words, reviews, now, args.fast_seconds)
        recall = {}
        if args.command != "train" and args.model:
            recall = probabilities(schedule, reviews, now, load_model(args.model))
        if args.command == "train":
            current = [r for r in reviews if r.word_id in known_ids and r.timestamp <= now]
            model = train(current, args.model)
            evaluation = model["evaluation"]
            print(f"\nChronological evaluation: {evaluation['training_samples']} train / {evaluation['test_samples']} test")
            for name in ("constant_baseline", "logistic_regression"):
                values = evaluation[name]
                print(f"  {name}: Brier {values['brier_score']:.4f} | log loss {values['log_loss']:.4f}")
            print("Lower Brier score and log loss are better. This does not demonstrate improved retention.")
            print(f"Saved all-data refit for future predictions: {args.model}")
        elif args.command == "stats":
            rows = show_stats(schedule, reviews)
            if args.export:
                export_stats(args.export, rows)
                print(f"Exported: {args.export}")
        elif args.command == "schedule":
            print("\nReview schedule (UTC; NEW means ready to learn):")
            for item in sorted(schedule, key=lambda item: (item.attempts == 0, item.due_at, item.word.id)):
                state = "NEW" if item.attempts == 0 else "DUE" if item.due_at <= now else "LATER"
                date = "now" if item.attempts == 0 else item.due_at.isoformat(timespec="minutes")
                estimate = recall.get(item.word.id)
                prediction = f" | estimated recall {estimate:.1%}" if estimate is not None else ""
                print(f"  {state:5} {item.word.french:16} {date} | {item.reason}{prediction}")
        else:
            if args.strategy == "random":
                items = random.Random(args.seed).sample(schedule, min(args.limit, len(schedule)))
            else:
                items = select_due(schedule, now, len(schedule))
                if args.strategy == "predictive":
                    # The model changes priority among due words. Rule-based
                    # dates still decide eligibility; unseen words come last.
                    items.sort(key=lambda item: (recall.get(item.word.id) is None,
                                                recall.get(item.word.id) if recall.get(item.word.id) is not None else 1,
                                                item.due_at, item.word.id))
                items = items[:args.limit]
            if not items:
                if schedule:
                    next_due = min(item.due_at for item in schedule)
                    print(f"Nothing due. Next review: {next_due.isoformat(timespec='minutes')} (UTC).")
                else:
                    print("No vocabulary found. Add rows to words.csv.")
            else:
                run_quiz(items, reviews_path, args.strategy)
    except (DataError, OSError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
