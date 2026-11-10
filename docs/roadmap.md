# Roadmap

Six steps, in order. Each one says where it stands today and the check that will show it is done. Today's numbers come from [results](results.md).

| # | Step | Today | Done when |
|---|---|---|---|
| 1 | **Real shot labels from a coach** | 96 hits on 5 rallies, labelled by an AI agent looking at the frames | A coach has marked who hit, when, the shot name and where it landed, on rallies kept apart for testing, and the hit scores are re-run against those labels |
| 2 | **Find 80% of hits**, then name the shots | 35% of the hits it reports match a label; it finds 33% of labelled hits. Names a shot on 6 of 336 hits; shot names never scored | At least 80% of its hits are right and it finds at least 80% of hits, on the coach's test rallies. Then shot names are scored the same way |
| 3 | **Pick the exciting moments automatically** | A fixed rule picks 3 clear, spaced-out moments. Excitement scores exist only in a prototype, set by hand | Two separate scores, "exciting" and "good to learn from", pick moments on matches they were not tuned on, and the picks agree with a person's picks more often than the fixed rule does |
| 4 | **Bring shuttle tracking into the app** | Studio only. Found in 233 of 282 frames of one 9-second stretch, not checked against hand marks | The app tracks the shuttle in uploaded clips, the full analysis still fits the 10-second limit, and positions are checked against hand-marked frames |
| 5 | **A person listens** to every narrated review | Machine checks only: audio matches the plan, speech-to-text found no dropped sounds | Each review has a listening record (who, when, what was wrong), and a review is only called finished after it |
| 6 | **Run and time it on an M4 Pro MacBook** | Built and timed on Windows 11 with a 16-thread server CPU. Never run on a Mac | The tests pass on the Mac, and `fast/acceptance.py` records its times and accuracy there against the same 10-second limit |

Not on this list yet: the fast preview (3.59 s against a 1-second goal, [`turbo/SPEED.md`](../turbo/SPEED.md)).
