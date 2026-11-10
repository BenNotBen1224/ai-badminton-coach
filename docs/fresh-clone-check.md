# Fresh-clone check

Everything below was run on a clean clone of this repo, following [Use it](../README.md#use-it) and [reproducibility](reproducibility.md). The models and clips came from the repo's download scripts. `ffmpeg.exe` and `ffprobe.exe` were copied into `tools\`, as the docs ask.

**Computer:** AMD EPYC 7763 server CPU (8 cores, 16 threads), 64 GB memory, Windows 11 (build 26300), Python 3.12.10. This is the same kind of computer the [results](results.md) were measured on.

## Steps

| Step | Result |
|---|---|
| `git clone`, then `scripts\get_models.py` | Models in place. The two detectors and the voice model were hashed again and match the values in the script |
| `scripts\get_footage.py` | Clips 04–10 cut from the public match video. Clip 08 is `08-china-open-ws-f`. Its frames differ from the lab's clip 08 by 0.67–0.89 grey levels on average, against 5–51 for `ws-e`, the clip an older version of the script cut by mistake |
| `scripts\verify_manifest.py` | 442 of 442 files match [`results/MANIFEST.json`](../results/MANIFEST.json) |
| `scripts\run_tests.py` | 729 ran, 0 failures, 8 skipped, about 110 s |
| `python -m coach.complete_review` on clip 04 | Finished, exit code 0. See below |
| `fast\acceptance.py`, clips 04, 05 and 08 | Accuracy matches the records. Speed mostly over the 10 s limit. See below |

`fast\prepare_cache.py` was run after the tests and the review, before the third timing round. The first two timing rounds ran without it and were no slower.

### Tests

| Suite | Ran | Skipped | Seconds |
|---|---|---|---|
| coach | 573 | 7 | 92.9 |
| core | 74 | 0 | 8.2 |
| turbo | 71 | 1 | 7.3 |
| comparison | 7 | 0 | 0.8 |
| studio-tactics | 4 | 0 | 0.2 |

Why the 8 skip: [reproducibility](reproducibility.md#tests).

## Narrated review

From `outputs\my-review\run.json`:

| Check | Value |
|---|---|
| Video length | 73.87 s, 2,216 frames, audio the same length |
| Source frames kept | All 900, in order |
| Analysis (players, body points, events) | 11.8 s, 1,802 player crops |
| Whole command, including speech and video | 239.5 s |
| Narration | 5 Chinese segments with cautious wording, for example 投影肘角约 161 度 ("projected elbow angle about 161°") and 触球未确认 ("contact not confirmed") |

## Speed and accuracy again

### Accuracy

| Clip | Players found | Joints within 5% of player height | Recorded |
|---|---|---|---|
| 04 | 100% | 95.29% | 95.34% |
| 05 | 100% | 94.26% | 94.35% |
| 08 | 99.33% | 94.00% | 93.95% |

The small gaps come from the re-cut clips, which are not byte-identical to the measured ones. On the lab's original clip 04, the same code gave exactly the recorded 95.34%.

### Speed

35 timed runs on clips 04, 05 and 08: **9.63 s min, 10.17 s median, 12.52 s max**. Only 10 were under the 10 s limit. The recorded runs took 7.26–9.87 s.

Before each run, the CPU was a median of 20% busy, against 4% for the recorded runs. With nothing being timed, other apps alone used about 9%, mostly the desktop app's browser panels.

To rule out the repo itself, each difference from the lab setup was swapped back and timed on clip 04, alternating the two sides:

| Swap | Runs each | Lab side | Repo side |
|---|---|---|---|
| `fast/pipeline.py`: timed version vs published | 4 | 9.78–10.02 s | 9.63–10.64 s |
| Clip 04: lab original vs re-cut | 4 | 9.77–10.02 s | 9.63–10.64 s |
| Model bundle: lab's vs freshly prepared | 3 | 10.08–10.37 s | 9.91–10.92 s |
| Python environment: lab's vs fresh install | 3 | 10.49–10.77 s | 10.57–11.96 s |

No swap moved the time by more than run-to-run noise. The slowdown is the busier computer, not the code. So the 10 s limit holds only on a quiet computer.

## After this check

The repo was then rebuilt by the same script with documentation changes only. Every file outside `docs/`, `README.md` and the manifest itself has the same hash in `results/MANIFEST.json` as in the clone that was checked.
