# Vocabulary and review history

`words.csv` is the vocabulary source. Its columns are `id`, `french`, and
`english`. IDs are stable strings that link words to historical answers;
changing row order is safe, but reusing an ID for a different word is not.
Accepted English alternatives are separated with `|`, such as `house|home`.
CSV quoting is formatting and does not change the accepted translations.

`reviews.csv` is created when an answer is recorded. Each row contains
`word_id,timestamp,result,response_time`: result is 1 for correct or 0 for
incorrect, and response time is measured in seconds. New timestamps include
a UTC offset. The loader also accepts the original four-column headerless
log; older timestamps without an offset use the computer's local timezone.

Skipping or quitting does not record an attempt. Schedules are calculated
from this history rather than saved here. Keep a backup before importing
old logs, preserve their original vocabulary IDs, and keep personal review
history out of Git unless you deliberately intend to share it.
