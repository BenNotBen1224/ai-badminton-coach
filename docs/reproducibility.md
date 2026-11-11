# Reproducibility

What you can re-run from a clean clone, what needs downloads or clips you add by hand, and what is a record only.

| Level | What | Needs |
|---|---|---|
| 🟢 Re-runs | Tests, the app, the narrated review, speed and accuracy on clips 04, 05 and 08 | `get_models.py`, `prepare_cache.py`, `get_footage.py`, ffmpeg |
| 🟡 Needs your own copy | Speed and accuracy on clips 01–03; the 8 skipped tests | Clips or files that are not shipped (below) |
| 🟢 Re-scores frozen predictions | All tactics V3 variants, all five rallies, both splits | Full labels, predictions, and partition are shipped. No models or video needed |
| 🟢 Rebuilds media from frozen inputs | All three advertised short reels | Public source match, narration release bundle, ffmpeg, Chinese font |
| ⚪ Record only | The milestone-1 replay, the fast-preview time | Earlier run folders or original clip 03 |

## Re-score the tactics predictions

This command uses only Python's standard library. It does **not** run detection or pose inference.

```powershell
python studio\tactics\score_v3.py --out outputs\tactics-rescore --check-reference
```

`summary.json` must equal every historical metric in `studio/tactics/v3_eval.json` before the command succeeds. It also records the actual input hashes. The `as_emitted` result is 32 matches, 91 predictions within labelled windows, and 96 labels. Precision is 32/91, recall is 32/96, and zone agreement is 7/10. The 80% recall goal still fails.

The complete 336-row hit table, AI-authored labels, and development/evaluation partition are in `studio/tactics/inputs/`. Only local paths were scrubbed. The original and published hashes are in [`recovered-inputs.json`](../studio/reels/recovered-inputs.json). `v3_hits_sha256` names the historical unsanitized file; `input_sha256.hits` names the shipped bytes.

To score changed inputs, pass `--hits`, `--labels`, or `--partition` and omit `--check-reference`. The tests change both predictions and labels and require the scores to change. Missing inputs fail. These labels are not human ground truth, and the evaluation split was not blind. Matching uses the same hitter within 0.25 seconds, one-to-one, inside the labelled windows only. No claim of whole-match accuracy follows.

## Rebuild the short reels

This rebuilds fixed editorial choices for one match. It does not choose highlights, name shots, or write the approved commentary style for a new video. The separate teaching CLIs need the full app setup. `coach.short_review` passed a fresh clip-04 export and encoded-output recheck. `coach.editorial_short` rejected narration that did not fit its selected footage. See the [TikTok-style workflow status and command record](tiktok-style-workflow.md).

Use Python 3.12 and ffmpeg/ffprobe on `PATH` or in `tools\`. This path needs no detector weights and does not import an old base MP4. The default Chinese font is Microsoft YaHei on Windows. You can supply `--font` with another Chinese TrueType or OpenType font; the layout may differ. Only Windows has been exercised.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reels.txt
.\.venv\Scripts\python.exe scripts\get_footage.py --no-cut
.\.venv\Scripts\python.exe scripts\get_reel_inputs.py
.\.venv\Scripts\python.exe scripts\with_lock.py "render reels" -- .\.venv\Scripts\python.exe scripts\render_reels.py --edition technical --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\technical
.\.venv\Scripts\python.exe scripts\with_lock.py "render reels" -- .\.venv\Scripts\python.exe scripts\render_reels.py --edition base --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\base
.\.venv\Scripts\python.exe scripts\with_lock.py "render reels" -- .\.venv\Scripts\python.exe scripts\render_reels.py --edition coaching --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\coaching
```

The narration bundle is `reel-inputs-v1.zip` in release **v0.1.1**, separate from the historical MP4s in **v0.1.0**. Its URL, byte count, archive hash, individual WAV hashes, spoken text, and slots are in [`inputs-manifest.json`](../studio/reels/inputs-manifest.json). To use an already-downloaded bundle offline, run `get_reel_inputs.py --archive PATH_TO_ARCHIVE --out work\reel-inputs`. The script verifies the same hashes and exact member list in either case. No lab folder is needed.

Each output folder contains `reel.mp4`, `mix.wav`, a frame map, preview pictures, `expected-frames.npz`, `render-inputs.json`, and `verification.json`. The renderer decodes every new output frame, checks geometry and duration, checks visible captions against a no-caption control, and checks every voice slot against the encoded mix. The technical edition also compares every card frame with wrong-card controls. Failures raise an error; an existing output folder is never overwritten.

To recheck an encoded output without encoding it again, keep its original code, font, and inputs. The command enforces the saved generation hashes and checks that its expected frames, mix, and frame map have not changed. It reports original renderer and current verifier hashes separately. `-X utf8` also supports Chinese JSON output when you redirect it through a Windows pipe.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\render_reels.py --edition technical --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\technical --verify-only
.\.venv\Scripts\python.exe -m pytest tests\test_reproduction.py studio\tactics\test_eval_strict.py -q
```

Prove that the checks reject a short missing white subtitle and a muted voice line:

```powershell
.\.venv\Scripts\python.exe scripts\with_lock.py "negative controls" -- .\.venv\Scripts\python.exe scripts\check_reel_controls.py --render outputs\technical --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\negative-controls
```

This makes deliberately altered MP4s and requires `--verify-only` to reject them for the intended caption and audio failures. It does not change the original render.

The base and technical editions retain 798 frames at 30 fps, 1080 by 1920, lasting 26.6 seconds. Their source spans are 57.8–64.8, 126.6–134.0, and 272.8–285.0 seconds. Game 1 has Wang/red near and Han/white far. In the last span, game 2 has Han/white near and Wang/red far. Skeletons use the shipped frame-bound pose cache. The older coaching edition retains its 940-frame schedule, about 31.333333 seconds, and recuts source clip 04 at 20 seconds from the full match.

**Frozen narration is not new speech synthesis.** The bundle preserves the original technical and base WAV takes. The older coaching renderer discarded its temporary raw WAVs, so this bundle contains newly synthesized MeloTTS equivalents from the original caption text. They are marked as such. It contains no final MP4 or mixed court/voice track.

To synthesize speech again instead of using those takes, download the voice models and use a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts\get_models.py kokoro melo
.\.venv\Scripts\python.exe scripts\with_lock.py "new speech" -- .\.venv\Scripts\python.exe scripts\render_reels.py --edition technical --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\technical-new-speech --voice-mode synthesize --models models
```

Kokoro uses speaker 60 at speed 1.0; the coaching edition uses MeloTTS at speed 1.08. New speech must fit the unchanged slots or the command fails. Native TTS warnings remain visible. No human listening or voice-quality approval is implied.

**Regenerated equivalents are not exact-byte reference videos.** Source decoding, recutting clip 04, fonts, speech synthesis, and encoders can change bytes. The exact historical MP4 hashes stay attached to their original release. Rebuilding does not claim an identical MP4, rerun pose inference, validate the shot names, or improve tactics recall.

For fresh inference rather than replaying pose estimates, install the full `requirements.txt`, fetch models, and prepare the model cache as in the README. A new complete review uses `python -m coach.complete_review VIDEO OUT`. A new full-match tactics run uses:

```powershell
.\.venv\Scripts\python.exe -m coach.tactics footage\china-open-2025-ws-final.webm outputs\tactics-new --segments sample-videos\china-open-segments.json --states sample-videos\china-open-score-states.json --ffmpeg tools\ffmpeg.exe --pose-python .venv\Scripts\python.exe
```

That is new inference, not a way to recreate the frozen V3 table byte for byte. It was not rerun as part of this package, and its scores can differ. The shuttle path has its own `studio\shuttle\run_tracknet.py` command after `get_models.py tracknet`; the shipped ball CSVs and `studio/beat-facts.json` preserve the commentary's model evidence.

Footage attribution and the distinction between code rights and media rights are in [models and licences](models-and-licences.md#reproduction-inputs-and-result-videos).

The [clean-checkout record](reproduction-check.md) contains the executed commands, regenerated media hashes, negative controls, and machine-readable receipts.

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

Runs the five suites in the order below, each from the folder it expects, with `PYTHONPATH` set to the repo root. The initial `e27fb934` revision's clean-clone run with the models and clips 04–10 had **729 tests, 0 failures, and 8 skips** ([fresh-clone-check](fresh-clone-check.md)). This historical table does not include the new reproduction tests.

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
