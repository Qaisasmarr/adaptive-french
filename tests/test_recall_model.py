import math
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from models import Review, Word
from recall_model import build_dataset, features, fit, load_model, metrics, predict, train
from scheduler import build_schedule
from storage import DataError
from storage import append_review

ROOT = Path(__file__).resolve().parents[1]

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def synthetic_history():
    # Mechanically generated fixtures ONLY, never evidence about real learners.
    return [Review("1", NOW + timedelta(days=i), int(i % 3 != 0), float(i % 7 + 1)) for i in range(61)]


class RecallModelTests(unittest.TestCase):
    def test_features_do_not_use_the_current_attempt_outcome_or_time(self):
        first = Review("1", NOW, 1, 2)
        a = build_dataset([first, Review("1", NOW+timedelta(days=2), 0, 99)])
        b = build_dataset([first, Review("1", NOW+timedelta(days=2), 1, 1)])
        self.assertEqual(a[0][1], b[0][1])
        self.assertNotEqual(a[0][2], b[0][2])
        self.assertAlmostEqual(a[0][1][0], math.log1p(2))
        self.assertEqual(len(a), 1)

    def test_history_is_separate_for_each_word(self):
        records = [Review("1", NOW, 1, 2), Review("2", NOW+timedelta(days=1), 0, 10),
                   Review("1", NOW+timedelta(days=2), 1, 2)]
        dataset = build_dataset(records)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0][1][3], 1)
        self.assertEqual(dataset[0][1][2], math.log1p(1))

    def test_fit_learns_a_synthetic_signal_and_handles_constant_features(self):
        samples = [(NOW+timedelta(days=i), [float(i % 2), 1, 1, 1, 1], i % 2) for i in range(40)]
        model = fit(samples)
        self.assertGreater(predict(model, [1, 1, 1, 1, 1]), 0.8)
        self.assertLess(predict(model, [0, 1, 1, 1, 1]), 0.2)
        self.assertEqual(model["scales"][1:], [1, 1, 1, 1])

    def test_metrics_against_known_values(self):
        result = metrics([0, 1], [0.5, 0.5])
        self.assertAlmostEqual(result["brier_score"], 0.25)
        self.assertAlmostEqual(result["log_loss"], math.log(2))
        self.assertAlmostEqual(result["accuracy_at_0_5"], 0.5)

    def test_train_uses_later_holdout_and_saved_model_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model = train(synthetic_history(), path)
            self.assertEqual(model["fit_samples"], 60)
            evaluation = model["evaluation"]
            self.assertEqual(evaluation["training_samples"], 48)
            self.assertEqual(evaluation["test_samples"], 12)
            self.assertLess(evaluation["train_last_utc"], evaluation["test_first_utc"])
            self.assertEqual(load_model(path), model)
            with self.assertRaises(FileExistsError):
                train(synthetic_history(), path)

    def test_training_guard_for_small_and_single_class_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            for records in (synthetic_history()[:10],
                            [Review("1", NOW+timedelta(days=i), 1, 2) for i in range(61)]):
                with self.assertRaises(DataError):
                    train(records, path)
                self.assertFalse(path.exists())

    def test_invalid_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text('{"schema_version": 999}')
            with self.assertRaises(DataError):
                load_model(path)

    def test_equal_timestamps_never_straddle_split(self):
        records = synthetic_history()[:49]
        records += [Review("1", NOW+timedelta(days=49), i % 2, 2) for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            model = train(records, Path(directory)/"model.json")
            self.assertLess(model["evaluation"]["train_last_utc"], model["evaluation"]["test_first_utc"])

    def test_model_commands_and_predictive_quiz_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "words.csv").write_text("id,french,english\n1,maison,house\n")
            for review in synthetic_history():
                append_review(data / "reviews.csv", review)
            model_path = data / "model.json"
            def run(command, *args, answers=None):
                return subprocess.run([sys.executable, str(ROOT / "quiz.py"), command,
                                       "--data-dir", str(data), "--model", str(model_path), *args],
                                      input=answers, text=True, capture_output=True)
            result = run("train")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("48 train / 12 test", result.stdout)
            result = run("schedule")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("estimated recall", result.stdout)
            result = run("quiz", "--strategy", "predictive", answers="house\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Correct answers: 1/1", result.stdout)

    def test_train_command_does_not_create_fake_model_with_no_history(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "words.csv").write_text("id,french,english\n1,maison,house\n")
            output = data / "model.json"
            result = subprocess.run([sys.executable, str(ROOT / "quiz.py"), "train",
                                     "--data-dir", str(data), "--model", str(output)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("found 0", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
