# TikTok-style reel workflow

The repo can rebuild the approved Han Yue versus Wang Zhiyi commentary reel. It does not have a command that watches a new match, learns a commentary style, picks highlights, names the shots, and writes that reel automatically.

<img src="images/tiktok-workflow.svg" alt="Six steps. Style learning is not shipped. Moment selection, shot names, and title and lines are hand-edited for the reference match. Rendering and encoded-output verification work for that match." width="100%">

## What exists

| Path | What it does | Evidence and limit |
|---|---|---|
| [`scripts/render_reels.py`](../scripts/render_reels.py) | Rebuilds the technical commentary, base, and earlier coaching editions from source frames, shipped pose estimates, and narration inputs | [Clean-checkout proof](reproduction-check.md), including freshly synthesized technical speech. The source hash, selections, player mapping, and scripts belong to this match |
| [`coach.short_review`](../coach/short_review.py) | Runs fresh Fast pose analysis and selects three supported teaching examples for a vertical short | Fresh clip-04 export passed. Its hook is fixed to `这一跨，学三个关键`. This is a technique lesson, not the approved match-commentary format |
| [`coach.editorial_short`](../coach/editorial_short.py) | Runs fresh Fast analysis for a legs, core, tactics, and arms breakdown | Fresh clip-04 export failed. Its legs narration exceeds its selected footage. This is not a generator of named-shot commentary |
| [`coach.verify_short`](../coach/verify_short.py) | Rechecks the teaching short against its source and saved pose evidence | Passed on the freshly generated technique short. It samples 10 encoded pictures and every planned caption's audible audio. It is not the all-frame commentary checker or an editorial-short verifier |
| [`studio/editorial/viral-v3/run_render.py`](../studio/editorial/viral-v3/run_render.py) and [`studio/commentary/`](../studio/commentary/) | Preserve the earlier editing and commentary code | Historical entry points retain dependencies on old run folders. Use `scripts/render_reels.py` for the supported reproduction command |

The technique exporters are CLI-only. The app exports a continuous narrated review. Neither the app nor those exporters generates the approved commentary style for a new video.

## What the approved script shows

These are editorial choices visible in the shipped [technical script](../studio/reels/technical-script.json), not rules learned or enforced by a style model.

- Name the players. The title names Han Yue and Wang Zhiyi, and spoken lines identify who attacks or defends.
- Use specific badminton terms rather than only describing visible movement. Examples include `反手斜线拦截`, `底线暴力跳杀`, `接杀走对角`, and `杀吊结合`.
- Keep speech short. The final script replaces longer explanatory paragraphs with brief shot calls and reactions such as `神仙防守！`. It does not have an automatic test for obvious or unhelpful commentary.
- Put measured numbers in on-screen tags, not in the spoken lines. For example, `韩悦底线暴力跳杀！` is spoken while `跳杀 · 约 0.5 秒到接杀` appears as a tag.

The moment choices, title, player names, shot terms, and lines were edited for this match. Shot names came from frame review, not a validated shot classifier. Timing and metres are model estimates. The script does not prove that a reel will attract viewers.

## What the six steps mean today

| Step | Current state |
|---|---|
| Learn style | Not shipped. The script preserves one approved example, not a style-learning workflow |
| Pick moments | The commentary uses three hand-selected source spans. Teaching exporters have different evidence-based selection rules |
| Name shots | The commentary's terms came from frame review. Tactics recall still fails 80%, and automatic shot naming is not established |
| Write title and lines | The title, lines, caption text, and tags are shipped as fixed JSON. No generator writes them for a new match |
| Render with voice | Rebuilding all three advertised editions works for this match. Technical fresh-speech synthesis also passed. Outputs are 9:16 with captions and skeletons |
| Verify | Commentary reproduction checks decoded frames, planned captions, cards, and voice coverage. Missing short text and muted narration are rejected. These checks do not approve shot names or voice naturalness |

## Rebuild or use a teaching exporter

To rebuild the approved example, follow [Rebuild the short reels](reproducibility.md#rebuild-the-short-reels). Exact historical MP4s remain in [release v0.1.0](../results/README.md). Rebuilt equivalents and fresh speech are not claimed to have the same bytes.

For fresh technique analysis on a supplied clip, use the separate [teaching exporter commands](../coach/README.md#vertical-technique-short-experimental). They require the full app dependencies, detector models, prepared Fast cache, local MeloTTS voice, and source attribution. Do not treat those commands as automatic commentary production or as validated on arbitrary footage.

## Fresh-clip command check

The [machine-readable receipt](../results/reproduction/any-clip-commands.json) records commands, source and output hashes, fresh-inference timings, and decoded-output checks. It started from clean revision `10873df`, installed the full declared app requirements, exported both public YOLO models with matching recorded hashes, prepared the Fast cache, and recut clip 04 through `get_footage.py`. Model export also installed its missing optional `onnxruntime` dependency.

```powershell
python -X utf8 -B -m coach.short_review sample-videos\04-china-open-ws-a.mp4 outputs\any-clip-short-fresh
python -X utf8 -B -m coach.verify_short outputs\any-clip-short-fresh sample-videos\04-china-open-ws-a.mp4
python -X utf8 -B -m coach.editorial_short sample-videos\04-china-open-ws-a.mp4 outputs\any-clip-editorial-fresh
```

The technique short passed. Its MP4 has 938 frames, 31.266667 seconds, 1080 by 1920 pixels, and SHA256 `2cfe0f59128e2711f06bab595e879bbdc8054a70d4e434db3005f5784557087a`. The run used fresh Fast inference on 900 input frames, not a stored pose replay. The encoded recheck reproduced the selected evidence, checked all 10 planned picture samples and caption audio windows, and confirmed H.264, AAC, faststart, and no selected excerpt crossing a cut.

The editorial exporter failed before encoding. Its first failure requested an initial court anchor at frame 53, outside its sampled range of frames 60 through 210, even though frame 53 had measured ankles. The bounded correction includes that requested frame in the scan. A regression fails on the original code and passes with the correction. No confidence gate or fallback changed.

A fresh editorial retry with that correction then failed because the legs narration did not fit the selected forward footage. The existing editorial test class also stops at that speech-fit guard on its stored pose. Neither a full editorial test pass nor a new editorial MP4 is claimed. The guard remains intact. Fixing the story and speech timing requires a separate change.

Single functional wall times were 88.14 seconds for technique export and 6.34 seconds for its recheck. The failed editorial attempts took 13.60 and 27.12 seconds. These include startup and validation on a shared 16-logical-CPU host. They are execution receipts, not speed benchmarks or estimates of arbitrary-video runtime.

No human listening or naturalness review is claimed. Media attribution and the caveat about uncleared upstream broadcast rights still apply.
