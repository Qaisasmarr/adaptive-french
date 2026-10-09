import csv
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import summarize
from models import Review, Word
from quiz import main
from scheduler import DAY, MAX_INTERVAL, build_schedule, select_due
from storage import DataError, append_review, is_correct, load_reviews, load_words

NOW = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
WORD = Word("1", "maison", "house|home")


class SchedulerTests(unittest.TestCase):
    def test_new_words_ready_without_made_up_history(self):
        schedule = build_schedule([WORD], [], NOW)
        self.assertEqual(select_due(schedule, NOW, 10), schedule)
        self.assertEqual(schedule[0].attempts, 0)

    def test_failure_due_after_ten_minutes_including_boundary(self):
        review = Review("1", NOW, 0, 2)
        schedule = build_schedule([WORD], [review], NOW)
        self.assertEqual(select_due(schedule, NOW + timedelta(minutes=9), 10), [])
        self.assertEqual(len(select_due(schedule, NOW + timedelta(minutes=10), 10)), 1)

    def test_fast_success_intervals_and_cap(self):
        history = []
        for index, expected in enumerate([DAY, 3*DAY, 6*DAY, 12*DAY, 24*DAY, MAX_INTERVAL, MAX_INTERVAL]):
            history.append(Review("1", NOW + timedelta(days=index), 1, 2))
            self.assertEqual(build_schedule([WORD], history, NOW)[0].interval_minutes, expected)

    def test_slow_success_grows_less_and_failure_resets(self):
        history = [Review("1", NOW, 1, 2), Review("1", NOW + timedelta(days=1), 1, 9)]
        self.assertEqual(build_schedule([WORD], history, NOW)[0].interval_minutes, 1.5*DAY)
        history.append(Review("1", NOW + timedelta(days=2), 0, 3))
        self.assertEqual(build_schedule([WORD], history, NOW)[0].interval_minutes, 10)
        history.append(Review("1", NOW + timedelta(days=3), 1, 2))
        self.assertEqual(build_schedule([WORD], history, NOW)[0].interval_minutes, DAY)

    def test_oldest_due_known_words_before_new_and_no_future_words(self):
        words = [WORD, Word("2", "chien", "dog"), Word("3", "chat", "cat"), Word("4", "eau", "water")]
        history = [Review("1", NOW - timedelta(days=3), 1, 2),
                   Review("2", NOW - timedelta(days=2), 1, 2), Review("3", NOW, 1, 2)]
        queue = select_due(build_schedule(words, history, NOW), NOW, 3)
        self.assertEqual([item.word.id for item in queue], ["1", "2", "4"])

    def test_out_of_order_input_is_replayed_chronologically(self):
        history = [Review("1", NOW, 0, 2), Review("1", NOW - timedelta(days=2), 1, 2)]
        schedule = build_schedule([WORD], history, NOW)[0]
        self.assertEqual(schedule.interval_minutes, 10)
        self.assertEqual(schedule.due_at, NOW + timedelta(minutes=10))


class StorageAndAnalyticsTests(unittest.TestCase):
    def test_answer_normalization_and_explicit_alternatives(self):
        self.assertTrue(is_correct("  HOME  ", WORD.english))
        self.assertTrue(is_correct("thank   you", "Thank You|thanks"))
        self.assertFalse(is_correct("housing", WORD.english))
        self.assertFalse(is_correct("", WORD.english))

    def test_original_log_round_trip_preserves_headerless_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            original = "1,2026-10-06T10:00:00,1,2.55"  # No trailing newline.
            path.write_text(original, encoding="utf-8")
            append_review(path, Review("1", NOW, 0, 4.128))
            reviews = load_reviews(path)
            self.assertEqual(len(reviews), 2)
            self.assertEqual(reviews[-1].response_time, 4.13)
            self.assertTrue(path.read_text().startswith(original))
            self.assertNotIn("word_id", path.read_text())
            self.assertIsNotNone(reviews[0].timestamp.utcoffset())

    def test_new_log_writes_one_header_and_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            append_review(path, Review("1", NOW, 1, 2))
            append_review(path, Review("1", NOW + timedelta(days=1), 1, 3))
            self.assertEqual(len(load_reviews(path)), 2)
            self.assertEqual(path.read_text().count("word_id"), 1)
            self.assertIn("+00:00", path.read_text())

    def test_malformed_reviews_rejected_instead_of_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            for record in ["1,bad-date,1,2", "1,2026-10-08T00:00:00,2,1",
                           "1,2026-10-08T00:00:00,1,nan", "1,2026-10-08T00:00:00,1,-1"]:
                with self.subTest(record=record):
                    path.write_text(record)
                    with self.assertRaises(DataError):
                        load_reviews(path)

    def test_duplicate_word_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.csv"
            path.write_text("id,french,english\n1,maison,house\n1,chien,dog\n")
            with self.assertRaises(DataError):
                load_words(path)

    def test_analytics_has_no_fake_accuracy_for_unseen_words(self):
        words = [WORD, Word("2", "chien", "dog")]
        reviews = [Review("1", NOW - timedelta(days=2), 0, 8), Review("1", NOW, 1, 2)]
        rows = summarize(build_schedule(words, reviews, NOW), reviews)
        self.assertEqual(rows[0]["accuracy_pct"], 50)
        self.assertEqual(rows[0]["avg_response_seconds"], 5)
        self.assertIsNone(rows[1]["accuracy_pct"])
        self.assertEqual(rows[1]["next_review_utc"], "")


class CommandTests(unittest.TestCase):
    def test_quiz_logs_only_answers_and_stats_export_works_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "words.csv").write_text("id,french,english\n1,maison,house|home\n2,chien,dog\n3,chat,cat\n")
            def run(*args, answers=None):
                return subprocess.run([sys.executable, str(ROOT / "quiz.py"), *args, "--data-dir", str(data)],
                                      input=answers, text=True, capture_output=True, cwd=directory)
            result = run("quiz", "--limit", "3", answers=" HOME \n:skip\n:q\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Correct answers: 1/1", result.stdout)
            self.assertEqual(len(load_reviews(data / "reviews.csv")), 1)
            result = run("schedule")
            self.assertIn("LATER", result.stdout)
            self.assertIn("NEW", result.stdout)
            exported = data / "summary.csv"
            result = run("stats", "--export", str(exported))
            self.assertEqual(result.returncode, 0, result.stderr)
            with exported.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["accuracy_pct"], "100.0")
            self.assertEqual(rows[1]["accuracy_pct"], "")

    def test_eof_ends_cleanly_without_logging_an_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "words.csv").write_text("id,french,english\n1,maison,house\n")
            result = subprocess.run([sys.executable, str(ROOT / "quiz.py"), "--data-dir", str(data)],
                                    input="", text=True, capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("No answers recorded", result.stdout)
            self.assertFalse((data / "reviews.csv").exists())

    def test_random_selection_reproducible_and_explicitly_not_due_only(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "words.csv").write_text("id,french,english\n1,maison,house\n")
            append_review(data / "reviews.csv", Review("1", datetime.now(timezone.utc), 1, 2))
            def run(strategy):
                return subprocess.run([sys.executable, str(ROOT / "quiz.py"), "--data-dir", str(data),
                                       "--strategy", strategy, "--seed", "42"],
                                      input=":q\n", text=True, capture_output=True)
            self.assertIn("Nothing due", run("adaptive").stdout)
            self.assertIn("French: maison", run("random").stdout)
            self.assertEqual(run("random").stdout, run("random").stdout)

    def test_export_cannot_overwrite_input_or_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            words = data / "words.csv"
            original = "id,french,english\n1,maison,house\n"
            words.write_text(original)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["stats", "--data-dir", str(data), "--export", str(words)])
            self.assertEqual(words.read_text(), original)

    def test_empty_dataset_and_missing_word_file(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--data-dir", str(data)]), 1)
            (data / "words.csv").write_text("id,french,english\n")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--data-dir", str(data)]), 0)
            self.assertIn("No vocabulary", output.getvalue())


if __name__ == "__main__":
    unittest.main()
