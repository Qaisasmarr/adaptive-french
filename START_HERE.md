# Start here, Qais

Follow this walkthrough to run Adaptive French, inspect saved answers, and
trace the scheduling rules before training a recall model.

## 1. Run a five-word quiz

Open your cloned or extracted `adaptive-french` folder in VS Code, open its
terminal, and run:

```bash
python3 quiz.py quiz --limit 5
```

On Windows, replace `python3` with `py`. There are no packages to install.
Answer in English. Use `:q` to stop.

## 2. See what was saved

Open `data/reviews.csv`. The four columns are:

```text
word_id, timestamp, result, response_time
```

An answered question produces one row. `result` is 1 or 0. Skipping and quitting
do not produce rows. Your old headerless log also works.

## 3. See how the quiz adapts

```bash
python3 quiz.py schedule
python3 quiz.py stats
```

A wrong answer is due again after 10 minutes. A first correct answer is due
tomorrow. New words are ready immediately. Each word appears once per session.

Your original loop asked every row. This version builds a review schedule and
selects a subset before it starts asking questions.

## 4. Read one function first

Open `scheduler.py` and read `next_interval()`:

```python
if review.result == 0:
    return 10.0
```

This is the first adaptation rule. The function returns a number of minutes.
`build_schedule()` adds that interval to the latest answer's timestamp.
`select_due()` checks whether that date has arrived.

Trace these cases yourself:

| Previous interval | Answer | Time | New interval |
| --- | --- | --- | --- |
| No history | Wrong | 3 seconds | 10 minutes |
| No history | Correct | 3 seconds | 1 day |
| 1 day | Correct | 3 seconds | 3 days |
| 1 day | Correct | 8 seconds | 1.5 days |

## 5. Collect real history before training

Practice across several days. When you have enough repeat attempts:

```bash
python3 quiz.py train --model models/recall-v1.json
python3 quiz.py schedule --model models/recall-v1.json
python3 quiz.py quiz --strategy predictive --model models/recall-v1.json
```

Training needs at least 40 repeated-word examples and a mix of correct and
incorrect answers. Do not intentionally create wrong answers just to satisfy
the guard. More diverse real history is what makes this useful.

The model learns from earlier reviews and evaluates on later reviews. Its
evaluation compares it to a constant-probability baseline. Lower error is
better; an improvement is not guaranteed. The saved model is then refitted on
all available data for future practice.

## Keep your existing data safe

Back up your original folder. Preserve the original word IDs and vocabulary
when bringing reviews into this version; IDs connect the two files. Do not
combine your old reviews with sample words whose IDs mean something different.

See `README.md` for commands, algorithm details, limitations, tests, and the
next steps toward a data analytics portfolio project. Start by understanding
the scheduler; the model can wait until you have real history.
