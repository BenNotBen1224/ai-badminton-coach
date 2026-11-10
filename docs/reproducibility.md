# Reproducibility

What you can re-run from a clean clone, what needs downloads or clips you add by hand, and what is a record only.

| Level | What | Needs |
|---|---|---|
| 🟢 Re-runs | Tests, the app, the narrated review, speed and accuracy on clips 04, 05 and 08 | `get_models.py`, `prepare_cache.py`, `get_footage.py`, ffmpeg |
| 🟡 Needs your own copy | Speed and accuracy on clips 01–03; the 8 skipped tests | Clips or files that are not shipped (below) |
| ⚪ Record only | The short reels, the tactics scores, the milestone-1 replay, the fast-preview time | Lab files and labels that are not in this repo |

## Set up

Follow [Use it](../README.md#use-it) in the README: make the `.venv`, install `requirements.txt`, then run `scripts\get_models.py`, `fast\prepare_cache.py` and `scripts\get_footage.py`. Every download is checked against the hash recorded when the project was built.

Then check that nothing in the repo has changed:

```powershell
.\.venv\Scripts\python.exe scripts\verify_manifest.py
```

## Tests

```powershell
.\.venv\Scripts\python.exe scripts\run_tests.py
```

Runs the five suites in the order below, each from the folder it expects, with `PYTHONPATH` set to the repo root. On a clean clone with the models and clips 04–10: **729 ran, 0 failures, 8 skipped** ([fresh-clone-check](fresh-clone-check.md)).

| Suite | Ran | Skipped |
|---|---|---|
| coach | 573 | 7 |
| core | 74 | 0 |
| turbo | 71 | 1 |
| comparison | 7 | 0 |
| studio-tactics | 4 | 0 |

The 8 skipped tests each print why. Seven need something this repo does not ship; one is a placeholder that always skips:

| Suite | Why it skips |
|---|---|
| coach | Needs sample clip 01 and its stored body points |
| coach | Always skips by design: clip 05 is held out, and clip 01 has only four model-made contact references, too few to score a whole clip |
| coach | Needs `C2_FULL_EXPORT=1`, an opt-in flag for a slow full export |
| coach | Needs hand labels (`labels.json`) that were never finished |
| coach | Needs a lab hand-off file from an earlier review run |
| coach | Needs split-step labels to be declared ready |
| coach | Needs the original bytes of sample clip 04 (the recut is not byte-identical) |
| turbo | Needs sample clip 03, `tools\ffmpeg.exe` and the YOLO26 exports together |

## Speed and accuracy

The recorded runs are in [`fast/runs/`](../fast/runs/). `acceptance.py` counts frames with `tools\ffprobe.exe`, so copy `ffprobe.exe` into `tools\` first (it does not look on `PATH`). Then measure again on the clips `get_footage.py` makes:

```powershell
.\.venv\Scripts\python.exe fast\acceptance.py run --suite my-check --out my-check --runs 5 `
  --clip "04|sample-videos\04-china-open-ws-a.mp4|fast\runs\judge-04.json|fast\runs\baseline-04.json" `
  --clip "05|sample-videos\05-china-open-ws-b.mp4|fast\runs\judge-05.json|fast\runs\baseline-05.json" `
  --clip "08|sample-videos\08-china-open-ws-f.mp4|fast\runs\judge-08.json|fast\runs\baseline-08.json"
```

It writes `my-check\acceptance.json` with the same keys the [results](results.md) quote. Expect:

- **Accuracy close to, not equal to, the records.** The recut clips are not byte-identical to the ones that were measured.
- **Slower times on a busy computer.** The records were timed on a quiet 16-thread server CPU. On the same kind of computer with other apps running, the fresh-clone re-run took 9.6–12.5 s and mostly missed the 10 s limit ([fresh-clone-check](fresh-clone-check.md#speed-and-accuracy-again)). Run `fast\prepare_cache.py` once first, as the recorded runs did.
- Clips 01–03 need the original files from the Arise stream ([sample-videos](../sample-videos/README.md)).

## Narrated review

```powershell
.\.venv\Scripts\python.exe -m coach.complete_review sample-videos\04-china-open-ws-a.mp4 outputs\my-review
```

`run.json` in the output folder records the frame counts, timings and checks of that run. [fresh-clone-check](fresh-clone-check.md) has the run made from a clean clone.

## Records only

- **Short reels** ([`results/`](../results/)). They were rendered in the lab from the full match video with tools and scripts that are not all here. Each record keeps the video's hash and the checks of the finished file, so you can confirm a copy is the same video.
- **Tactics scores** ([`studio/tactics/v3_eval.json`](../studio/tactics/v3_eval.json)). Scoring needs the labels and the engine's full hit table, which are not shipped. [`results/tactics-v3-hits-summary.json`](../results/tactics-v3-hits-summary.json) has the counts and the hit table's hash.
- **Milestone-1 replay** ([milestone-1-pose-overlay](milestone-1-pose-overlay.md)). It needs run folders and comparison videos (about 45 MB) that are not published.
- **Fast preview, 3.59 s** ([`turbo/SPEED.md`](../turbo/SPEED.md)). Measured on the original clip 03.

## Code changes after timing

Two files differ from the ones in the run records' `code_sha256`:

- [`fast/pipeline.py`](../fast/pipeline.py) (`EEE0C7C6…` recorded, `F15E8D55…` published). If a video-reading thread fails, the error is now reported instead of being silently dropped. A test for that was added. Detection, body points and timing paths are unchanged. Timed side by side with the original on clip 04, it was no slower ([fresh-clone-check](fresh-clone-check.md#speed-and-accuracy-again)). If you pass `--expect-pipeline-sha` with the recorded hash, `acceptance.py` will refuse to run; leave it out.
- [`fast/acceptance.py`](../fast/acceptance.py). One line points to `tools\` at the repo root instead of the lab folder above it. It is the timer, not the timed code.

## Known gaps

- Windows 11 x64 only. Not run on a Mac.
- Accuracy is judged by a bigger model, and hits by AI-made labels, not people.
- Clips 01–03 and 11 cannot be fetched by script.
