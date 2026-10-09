# Adaptive French

A Python vocabulary trainer with review scheduling, learning analytics, and
an optional logistic regression recall model. It includes chronological
prediction evaluation. No real learner dataset or retention improvement is
claimed; training requires your actual review history.

## Why this project exists

The goal is to make French vocabulary practice respond to previous answers.
A quiz that asks every word in file order treats every word the same. This
project records correctness and response time, then uses that history to
decide which words are ready for another review. The intended benefit is more
focused practice; whether it improves long-term retention still needs testing.

It also connects a practical learning tool to a data analysis workflow:
collect attempts, summarize progress, and test whether past answers help
predict the next one. The rule-based scheduler works from the first session;
the optional recall model provides a way to explore prediction once enough
real history exists.

The main design choices support that goal:

- **Keep the raw attempts.** An append-only CSV preserves how each result
  happened, so schedules and summaries can be rebuilt as the code evolves.
- **Separate responsibilities.** Storage handles files, scheduling chooses
  review times, analytics summarizes history, and the quiz handles interaction.
  This makes each part easier to understand and change independently.
- **Start with readable rules.** Shorter intervals after mistakes and longer
  intervals after success make the scheduling decisions easy to trace.
- **Keep setup small.** Local files and the Python standard library let the
  project run without a server, account, or package installation.
- **Measure predictions honestly.** Training on earlier attempts and checking
  later ones tests a realistic direction of prediction. A constant baseline
  checks whether the extra model complexity is useful.

For a code walkthrough, follow `quiz.main()` → `load_words()` / `load_reviews()`
→ `build_schedule()` → `select_due()` → `run_quiz()` → `append_review()`.
The docstrings and nearby comments explain the reasons behind the key steps.

## Run it

Requires Python 3.9 or later. Uses the standard library only; no installation,
API key, or account is needed.

Extract the ZIP, open a terminal in the `adaptive-french` folder, then run:

```bash
# macOS / Linux
python3 quiz.py quiz
python3 quiz.py schedule
python3 quiz.py stats
```

On Windows, use `py` instead of `python3`:

```powershell
py quiz.py quiz
py quiz.py schedule
py quiz.py stats
```

The sample vocabulary contains 20 words. Answer in English. `:skip` leaves a
word unreviewed; `:q`, Ctrl+C, or end-of-input finishes the session. An empty
answer is an incorrect answer. Every answered question is saved immediately
in `data/reviews.csv`; the results use answered questions only.

```bash
python3 quiz.py quiz --limit 5
python3 quiz.py quiz --strategy random --seed 42 --limit 10
python3 quiz.py stats --export exports/word-summary.csv
python3 -m unittest discover -s tests -v
```

Export writes only to a new file, so choose a different filename on subsequent
exports. The seed controls word selection, not timings. The random practice
mode samples all words, including words not yet due, and updates the same
history. It is useful for practice and debugging; it is NOT an experiment that
can establish which scheduler produces better retention.

Paths to the default data directory are relative to `quiz.py`, so commands
also work when invoked from a different working directory. For your own data:

```bash
python3 quiz.py quiz --data-dir /path/to/your/adaptive-french/data
```

## Bring your existing data

This is a separate version based on your pasted quiz. It has not changed your
Mac folder or your GitHub repository. Back up your original folder before
moving files. You can point `--data-dir` at the old data folder, or copy its
`words.csv` and `reviews.csv` into this version's `data` folder.

**Keep your original word IDs and their meanings.** Review history is joined
to vocabulary by ID, not by row number. Replacing vocabulary with this sample
list while retaining old reviews can attach answers to the wrong word.

Your words file uses:

```csv
id,french,english
1,maison,house|home
2,chien,dog
```

The `|` separator means either answer is accepted. Matching ignores letter
case, leading/trailing whitespace, and repeated internal spaces. It does not
perform fuzzy or semantic matching. Add alternatives explicitly.

The original four-column headerless review log is supported:

```csv
1,2026-10-06T12:30:00,1,2.55
2,2026-10-06T12:30:10,0,6.81
```

New logs add the header `word_id,timestamp,result,response_time`. Existing
headerless logs stay headerless when appended. `result` is 1 for correct or 0
for incorrect. New timestamps include UTC offsets. Older timestamps without
an offset are interpreted in the current computer's local timezone. If those
records were created in a different timezone, correct them in a backed-up copy
before relying on exact due times. Invalid CSV records cause an explicit error
rather than silently losing data. Reviews for removed word IDs remain in the
file but are excluded from vocabulary statistics.

## How the scheduler works

At startup it reads the attempt log, groups attempts by word, orders them by
timestamp, and replays the rules below. The last attempt's timestamp plus its
resulting interval gives the next review time.

| Answer | Next interval |
| --- | --- |
| Incorrect | 10 minutes |
| First correct answer | 1 day |
| Correct and fast, previous interval = 1 day | 3 days |
| Correct and fast, previous interval > 1 day | Previous interval × 2 |
| Correct but slow | max(1 day, previous interval × 1.5) |

All intervals are capped at 30 days. A fast correct answer after a failure
restarts at 1 day. A slow first success also starts at 1 day.

"Fast" means at most 5 seconds by default. This is a configurable engineering
choice, not a scientifically established threshold:

```bash
python3 quiz.py quiz --fast-seconds 8
```

The configuration is applied when replaying all history. Changing the threshold
can therefore change existing due dates. Keep it fixed while collecting a
dataset for later evaluation. Typing speed, interruptions, and word length can
all affect measured response time. Time is measured from the English prompt
to submission and stored rounded to two decimal places.

Due, previously practiced words come first, ordered from oldest due date to
newest. Unseen words fill the remaining slots. Future reviews are excluded.
Each word appears at most once in a session: an incorrect answer becomes due
10 minutes later, rather than repeating immediately in the same session.

## Analytics

`stats` prints attempt count, overall accuracy, average response time, and up
to five words with the lowest historical accuracy. `--export` creates one row
per vocabulary item with counts, accuracy, response time, last review, next
review, and interval. Unseen words have blank accuracy rather than a fabricated
0% score. Tiny samples should not be used to label a word as mastered or weak.

These are descriptive statistics. Quiz accuracy does not measure delayed
retention, and differences between selected words do not prove a scheduler's
effectiveness. Raw logs support later analyses, but this version does not yet
record enough experimental metadata to claim a valid policy comparison.

## Train and evaluate a recall model

Collect reviews over multiple days first. Training needs at least 40 attempts
that have a previous attempt for the same word, at least 20 training examples,
and both correct and incorrect outcomes in the training period. This is a
minimum software guard, not a claim that 40 attempts make predictions reliable.

```bash
python3 quiz.py train --model models/recall-v1.json
python3 quiz.py schedule --model models/recall-v1.json
python3 quiz.py quiz --strategy predictive --model models/recall-v1.json
```

Use a new model filename for each training run. Without enough history, the
command explains what is missing and does not create a model. You can keep
using the rule-based quiz without training anything.

`recall_model.py` implements logistic regression using batch gradient descent
and L2 regularization with the Python standard library. It estimates the
probability of a correct answer from five features:

| Feature | Information available before the answer |
| --- | --- |
| Log elapsed days | Time since that word's previous review |
| Smoothed prior accuracy | `(prior correct + 1) / (prior attempts + 2)` |
| Log prior attempts | Number of earlier attempts for that word |
| Last correct | Whether the previous answer was correct |
| Log last response seconds | Previous response time, clipped to 300 seconds |

First-ever attempts are omitted from the model dataset because they have no
prior word history. Features are built before adding the current answer to
history. The earlier approximately 80% of examples train the evaluation model;
the later approximately 20% evaluate it. Equal timestamps stay on the same
side. Feature scaling is learned on the training period only.

Evaluation is **one-step-ahead**: each later test example can use outcomes from
earlier test examples, since those answers would already be known at that
moment in practice. It is not a multi-day forecast that withholds every test
outcome at once. No held-out outcome updates the evaluation model's weights.

The output compares Brier score (mean squared probability error) and log loss
against a constant probability equal to training accuracy. Lower scores are
better. Accuracy at a 0.5 threshold is also recorded in the model JSON. The
logistic model may lose to the constant baseline, especially with little data;
that is a useful finding to report honestly.

After evaluation, the command refits on all available samples and saves that
model for future practice. The recorded holdout scores belong to the earlier,
training-only fit. Predictive practice still uses the rule-based due dates,
then ranks due, practiced words by lowest estimated recall probability. New
words fill the remaining slots without fabricated probabilities. It does not
implement probability-derived due dates yet.

Predictions have not been calibrated or validated on your own history until
you collect and evaluate that data. This version pools word histories for one
learner; it does not model word identity, provide uncertainty intervals, or
enforce decreasing recall as time passes. Its learned time coefficient can
even have an implausible sign in a biased dataset. Selection policy, typing
speed, and practice timing can confound the results. Do not interpret a score
as proof of memory strength or improved retention.

The tests use small synthetic fixtures to check learning, serialization, and
split correctness. Their metrics are not portfolio results. The archive ships
with no fitted model, personal history, or fabricated learner dataset.

## Understand the code

| File | Responsibility |
| --- | --- |
| `quiz.py` | Commands, prompts, response timing, feedback, session results |
| `storage.py` | Read, validate, normalize, and append CSV records |
| `models.py` | `Word`, `Review`, and `Schedule` records |
| `scheduler.py` | Compute intervals and choose due words |
| `analytics.py` | Summaries and CSV export |
| `recall_model.py` | Pre-attempt features, logistic regression, held-out evaluation, and model files |
| `tests/test_project.py` | Scheduling boundaries, data compatibility, and command integration |
| `tests/test_recall_model.py` | Feature leakage checks, chronological splits, fitting, and model validation |

Read `scheduler.py` first. The core change from the original program is
`select_due(...)`: the quiz no longer blindly loops over every row in the
vocabulary file. It asks only the selected words.

This version separates data storage from scheduling so that a later model
can replace the rules without rewriting the quiz. History remains append-only
and local. Tests use temporary directories and never modify your real reviews.
It is designed for one learner and one running process; concurrent sessions
are not supported.

## Next milestones for an internship portfolio

1. **SQL and a trustworthy dataset.** Move to SQLite with migration of original
   CSVs, stable word IDs, sessions, learner IDs, policy versions, thresholds,
   elapsed time since previous review, and scheduled versus actual intervals.
2. **Analysis.** Plot delayed accuracy against elapsed time, inspect word
   difficulty and response-time distributions, and document missing data,
   sample size, and confounding factors. Keep fabricated/demo data clearly
   separate from real records.
3. **Improve prediction.** Run the supplied constant baseline and logistic
   model on sufficient real history. Add calibration analysis, stronger
   baselines, rolling chronological evaluation, and uncertainty estimates.
   Keep future reviews out of training and avoid tuning on the final test
   period. Do not call response-time thresholds a learned model.
4. **Scheduling evaluation.** With enough real data, predefine a comparison of
   fixed and adaptive scheduling, review budgets, randomization, and a delayed
   retention test. More correct answers during practice alone are not evidence
   of better retention. Offline prediction quality also does not establish
   that a scheduling policy causes an improvement.
5. **Interface and presentation.** Add an interactive dashboard after the
   data pipeline and evaluation work. Write an honest README with architecture,
   measured findings, limitations, reproducible commands, and a short demo.

You can currently describe this as a Python vocabulary trainer with heuristic
scheduling, descriptive analytics, and a logistic recall model with temporal
holdout evaluation. Personal-data validation, controlled retention evaluation,
a web dashboard, deployment, and any measured retention improvement remain
future work.

## First task to learn the implementation

Start one short quiz, inspect `data/reviews.csv`, and run `schedule`. Then
trace one record through `next_interval()` in `scheduler.py`: an incorrect
answer sets an interval of 10 minutes; its timestamp plus 10 minutes becomes
the next due date. Explain why `select_due()` excludes that word before that
time. Once that is clear, the next useful addition is SQLite storage.

## Version control

The ZIP includes source and tests, not a Git repository. When adding this code
to your existing repository, inspect the changes before committing. The supplied
`.gitignore` excludes future personal review logs and exports, but it cannot
untrack files that you already committed. Do not commit your actual learning
history unless you intentionally want it in that repository.
