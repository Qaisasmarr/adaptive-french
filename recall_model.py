"""Small logistic regression with chronological evaluation and no dependencies.

Features for an attempt are computed BEFORE that attempt enters history.
Prediction accuracy is distinct from the causal effect of a review policy.

Public API: train evaluates an early-period fit on later attempts, then saves
an all-data refit. probabilities estimates current recall for practiced words.
Predictions rank eligible words; the rule-based scheduler still sets due dates.
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from storage import DataError

FEATURE_NAMES = ["log_elapsed_days", "prior_accuracy_smoothed", "log_prior_attempts",
                 "last_correct", "log_last_response_seconds"]
MIN_SAMPLES = 40


def features(history, at):
    """Describe a word using prior reviews in chronological order.

    Log transforms compress large counts and delays; smoothing avoids treating
    one success as certain recall. Clipping response time limits the influence
    of long interruptions. No history means no supported prediction yet.
    """
    if not history:
        return None
    last = history[-1]
    days = max(0, (at - last.timestamp).total_seconds() / 86400)
    return [math.log1p(days), (sum(r.result for r in history) + 1) / (len(history) + 2),
            math.log1p(len(history)), float(last.result),
            math.log1p(min(last.response_time, 300))]


def build_dataset(reviews):
    """Pair pre-answer features with the answer they are meant to predict."""
    histories = defaultdict(list)
    samples = []
    for review in sorted(reviews, key=lambda r: r.timestamp):
        history = histories[review.word_id]
        row = features(history, review.timestamp)
        if row is not None:
            samples.append((review.timestamp, row, review.result))
        # Add the answer only after extracting features, or the model could
        # learn from the very outcome it is supposed to predict (data leakage).
        history.append(review)
    return samples


def sigmoid(value):
    """Convert a score to a probability without overflowing the exponential."""
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def fit(samples, iterations=800, learning_rate=0.1, regularization=0.01):
    """Fit logistic regression using batch gradient descent.

    Learn scaling only from the supplied samples to keep evaluation data out
    of training. A constant feature gets scale 1 to avoid division by zero.
    L2 regularization discourages large weights; the intercept is exempt.
    """
    count = len(samples)
    width = len(FEATURE_NAMES)
    means = [mean(row[1][i] for row in samples) for i in range(width)]
    scales = [math.sqrt(mean((row[1][i] - means[i]) ** 2 for row in samples)) or 1.0
              for i in range(width)]
    inputs = [[1.0] + [(row[1][i] - means[i]) / scales[i] for i in range(width)] for row in samples]
    weights = [0.0] * (width + 1)
    for _ in range(iterations):
        gradient = [0.0] * len(weights)
        for values, sample in zip(inputs, samples):
            error = sigmoid(sum(w*x for w, x in zip(weights, values))) - sample[2]
            for i, value in enumerate(values):
                gradient[i] += error * value
        weights = [weight - learning_rate * (gradient[i]/count + (regularization*weight if i else 0))
                   for i, weight in enumerate(weights)]
    return {"feature_names": FEATURE_NAMES, "means": means, "scales": scales, "weights": weights}


def predict(model, values):
    """Apply the saved training scales and weights to one feature vector."""
    scaled = [1.0] + [(value-center)/scale for value, center, scale
                     in zip(values, model["means"], model["scales"])]
    return sigmoid(sum(weight*value for weight, value in zip(model["weights"], scaled)))


def metrics(labels, probabilities):
    """Measure probability errors as well as correct/incorrect decisions."""
    clipped = [max(1e-12, min(1-1e-12, p)) for p in probabilities]
    return {
        "brier_score": mean((p-y)**2 for p, y in zip(probabilities, labels)),
        "log_loss": -mean(y*math.log(p)+(1-y)*math.log(1-p) for p, y in zip(clipped, labels)),
        "accuracy_at_0_5": mean(int((p >= 0.5) == bool(y)) for p, y in zip(probabilities, labels)),
    }


def train(reviews, output: Path):
    """Evaluate on later attempts, then save an all-data fit for future use.

    A chronological holdout mirrors predicting future practice from the past.
    Comparing with constant training accuracy checks whether learning feature
    weights adds value over always predicting the same probability.
    """
    samples = build_dataset(reviews)
    if len(samples) < MIN_SAMPLES:
        raise DataError(f"Need at least {MIN_SAMPLES} repeated-word attempts; found {len(samples)}. "
                        "Keep practicing over multiple days. This minimum is a software guard, not a guarantee of validity.")
    boundary = int(len(samples)*0.8)
    # Keep identical timestamps on the same side of the split.
    while boundary > 0 and samples[boundary-1][0] == samples[boundary][0]:
        boundary -= 1
    training, held_out = samples[:boundary], samples[boundary:]
    if len(training) < 20 or len(held_out) < 8:
        raise DataError("Need more distinct review times for a chronological 80/20 split.")
    if {row[2] for row in training} != {0, 1}:
        raise DataError("The training period must contain both correct and incorrect repeated-word attempts.")
    fitted = fit(training)
    baseline = mean(row[2] for row in training)
    labels = [row[2] for row in held_out]
    predicted = [predict(fitted, row[1]) for row in held_out]
    evaluation = {
        "split": "chronological 80/20; equal timestamps kept together",
        "test_protocol": "one-step-ahead: earlier test outcomes are available to later test features",
        "training_samples": len(training), "test_samples": len(held_out),
        "train_last_utc": training[-1][0].isoformat(), "test_first_utc": held_out[0][0].isoformat(),
        "constant_baseline_probability": baseline,
        "logistic_regression": metrics(labels, predicted),
        "constant_baseline": metrics(labels, [baseline]*len(held_out)),
    }
    # Save the all-data refit for future practice; held-out metrics above were
    # produced by the earlier training-only model, not this refit.
    final = fit(samples)
    final.update({"schema_version": 1, "model_type": "logistic_regression",
                  "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                  "fit_samples": len(samples), "evaluation": evaluation})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(final, file, indent=2, allow_nan=False)
        file.write("\n")
    return final


def load_model(path: Path):
    """Reject incompatible feature layouts and unusable model parameters."""
    try:
        with path.open(encoding="utf-8") as file:
            model = json.load(file)
        if model.get("schema_version") != 1 or model.get("model_type") != "logistic_regression":
            raise ValueError("unsupported model type/version")
        if model["feature_names"] != FEATURE_NAMES:
            raise ValueError("incompatible features")
        for name, length in (("weights", 6), ("means", 5), ("scales", 5)):
            values = model[name]
            if len(values) != length or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError(f"invalid {name}")
        if any(scale <= 0 for scale in model["scales"]):
            raise ValueError("scales must be positive")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise DataError(f"{path}: invalid model: {error}") from error
    return model


def probabilities(schedule, reviews, now, model):
    """Estimate current recall from history available by now, or None if new."""
    histories = defaultdict(list)
    for review in sorted(reviews, key=lambda r: r.timestamp):
        if review.timestamp <= now:
            histories[review.word_id].append(review)
    result = {}
    for item in schedule:
        values = features(histories[item.word.id], now)
        result[item.word.id] = predict(model, values) if values is not None else None
    return result
