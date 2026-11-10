# Clean-checkout reproduction

The reproduction package was run from a separate clone at `565e333bcae597e72e32afe274e56dbd4852e351`. Its tracked files were clean. A new Python 3.12 environment installed only `requirements-reels.txt`. No lab scripts, old MP4s, detector weights, or cached inference environments were used by the frozen-input render commands.

The source match was downloaded again with `scripts\get_footage.py --no-cut`. Its 205,677,737 bytes matched SHA256 `2500471d4852a4a73e0f4eb25ad28171b3c54967f58e1cc7620af16b46ccbb52`. The 19-file narration bundle was unpacked from the staged release asset with `get_reel_inputs.py --archive`, using the same hashes as the future public download.

## What ran

These commands ran inside the separate clone after installing ffmpeg and ffprobe in `tools\`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-reels.txt
.\.venv\Scripts\python.exe scripts\verify_manifest.py
.\.venv\Scripts\python.exe scripts\get_footage.py --no-cut
.\.venv\Scripts\python.exe scripts\get_reel_inputs.py --archive work\reel-inputs-v1.zip
.\.venv\Scripts\python.exe studio\tactics\score_v3.py --out outputs\tactics-rescore --check-reference
.\.venv\Scripts\python.exe scripts\render_reels.py --edition technical --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\technical
.\.venv\Scripts\python.exe scripts\render_reels.py --edition base --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\base
.\.venv\Scripts\python.exe scripts\render_reels.py --edition coaching --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\coaching
.\.venv\Scripts\python.exe scripts\check_reel_controls.py --render outputs\technical --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\negative-controls
.\.venv\Scripts\python.exe -m pytest tests\test_reproduction.py studio\tactics\test_eval_strict.py -q
```

Each heavy command ran under `scripts\with_lock.py` with the host's shared CPU lease. That lease coordinates other sessions, not reproduction inputs. On a standalone machine, the default repo-local lease shown in the [reproduction guide](reproducibility.md#rebuild-the-short-reels) is enough.

The optional fresh-speech route also ran in this checkout after downloading and hash-checking the public Kokoro and MeloTTS models:

```powershell
.\.venv\Scripts\python.exe scripts\get_models.py kokoro melo
.\.venv\Scripts\python.exe scripts\render_reels.py --edition technical --source footage\china-open-2025-ws-final.webm --inputs work\reel-inputs --out outputs\technical-new-speech --voice-mode synthesize --models models
```

All 10 new technical voice lines fit their original slots. The new MP4 passed the same 798-frame picture, card, caption, and voice checks. Its hash is `cb4a91e02882727cee140e3e2f00b822b7e462a7bb2a255107e5338d502d4895`. Kokoro printed an unknown-token warning from its bundled lexicon. No listening or speech-to-text approval follows from these checks.

## Results

All editions are 1080 by 1920 at 30 fps. Every decoded frame was compared with the newly generated picture. Each planned voice slot matched the encoded mix. Caption checks compare only pixels that distinguish the expected words from a same-frame control with those words removed.

| Edition | Frames checked | Duration | Caption frames checked | Voice slots |
|---|---:|---:|---:|---:|
| Technical | 798 | 26.6 s | 798 | 10 |
| Base | 798 | 26.6 s | 649 | 3 |
| Coaching | 940 | 31.333333 s | 850 | 6 |

Gaps in the base and coaching caption counts are planned silent caption gaps, not missing coverage. The technical edition additionally checked all 798 card frames against wrong-card controls.

The new technical MP4 hash is `755844efe3897305c0c2b7d2ca6d2306ec02e71b6a17c9bdb9aad4ef6718b067`. The base hash is `be7d0fec48bfd0ca7e09511b085e7ab22772066b1680df4826689fa1ac89805b`. The coaching hash is `54c15ddf36551f90ca7ee72efebf42cca32c2824ba49694afd946d2c0eb0cb67`. These are newly encoded equivalents, not the [historical release bytes](../results/reference-videos.json).

The renderer hash was `7a7c8773b7c4615df18eda6a41d8f4254a1738f9096079638b9000e98e65bc29`. Rechecks enforce saved source, code, font, pose, timeline, narration-manifest, expected-frame, mix, and frame-map hashes. Reports distinguish the original renderer from the current verifier.

All V3 metrics, all three scoring variants, and both development/evaluation splits matched the historical record. The emitted predictions still match 32 of 96 AI-labelled hits, below the 80% recall target.

The 12 focused tests passed. They include changed-label, changed-prediction, missing-input, wrong-source, changed-voice, archive-hash, and approved-wording checks. The media controls made actual altered MP4s. Removing only the short white `神仙防守！` line, while leaving its green tag and footer, failed the caption check. Muting its voice failed the audio check.

Machine-readable receipts and original generation bindings are in [`results/reproduction/`](../results/reproduction/). The original 729-test app run belongs to initial revision `e27fb934`, not this focused run.

A positive encoded-output recheck also passed. On Windows, redirecting its Chinese JSON output through a pipe required Python's `-X utf8` option. Without that option, the checks completed but printing to the legacy encoding failed. The [reproduction guide](reproducibility.md#rebuild-the-short-reels) includes the UTF-8 option.

Three additional recheck controls changed the saved original renderer hash, source hash, and mix hash in turn. Each failed for its intended binding error, and the original binding file was restored byte for byte.

The [decision log](reproduction-decisions.tsv) records input and verification choices. Its first five timestamps were reconstructed approximately, not captured from a runtime trace.

## Limits

No fresh pose, full-match tactics, or shuttle inference was run here. The labels remain AI-made. Shot names in the final commentary came from frame review, and times and metres are estimates. No human listening or naturalness review is claimed.

The original temporary coaching WAVs were not retained. The bundle contains equivalent MeloTTS takes synthesized from the original text and marks them as regenerated. The reference MP4s remain available byte for byte in release v0.1.0.

The code and small narration bundle do not transfer rights to upstream footage or broadcast content. See [attribution and licences](models-and-licences.md#reproduction-inputs-and-result-videos). Publication of the v0.1.1 bundle is a separate release-coordination step.
