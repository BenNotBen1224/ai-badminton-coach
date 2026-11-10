# TikTok-style reel workflow

The repo can rebuild the approved Han Yue versus Wang Zhiyi commentary reel. It does not have a command that watches a new match, learns a commentary style, picks highlights, names the shots, and writes that reel automatically.

<img src="images/tiktok-workflow.svg" alt="Six steps. Style learning is not shipped. Moment selection, shot names, and title and lines are hand-edited for the reference match. Rendering and encoded-output verification work for that match." width="100%">

## What exists

| Path | What it does | Evidence and limit |
|---|---|---|
| [`scripts/render_reels.py`](../scripts/render_reels.py) | Rebuilds the technical commentary, base, and earlier coaching editions from source frames, shipped pose estimates, and narration inputs | [Clean-checkout proof](reproduction-check.md), including freshly synthesized technical speech. The source hash, selections, player mapping, and scripts belong to this match |
| [`coach.short_review`](../coach/short_review.py) | Runs fresh Fast pose analysis and selects three supported teaching examples for a vertical short | Its hook is fixed to `这一跨，学三个关键`. This is a technique lesson, not the approved match-commentary format. The any-clip CLI has not yet been rechecked in the new reproduction environment |
| [`coach.editorial_short`](../coach/editorial_short.py) | Runs fresh Fast analysis for a legs, core, tactics, and arms breakdown | This is another teaching format, not a generator of named-shot commentary. The any-clip CLI has not yet been rechecked in the new reproduction environment |
| [`coach.verify_short`](../coach/verify_short.py) | Rechecks the teaching short against its source and saved pose evidence | It is separate from the all-frame commentary checks in `render_reels.py`. The [coach guide](../coach/README.md#vertical-technique-short-experimental) gives the commands |
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

No human listening or naturalness review is claimed. Media attribution and the caveat about uncleared upstream broadcast rights still apply.
