# Records

Records of finished renders and runs. **No video files are stored here**: each record holds the hashes, scripts, timings and checks of a video that was made, so you can see exactly what it contained and how it was checked.

| Folder or file | What it is |
|---|---|
| [`technical-commentary-v2/`](technical-commentary-v2/) | Latest vertical reel: Han Yue vs Wang Zhiyi (China Open 2025 women's final), 26.6 s, Chinese technical commentary with a Kokoro voice. `planned-schedule.json` is the script and timing; `verification.json` is the check of the finished file. |
| [`editorial-base-v3/`](editorial-base-v3/) | The reel the commentary was laid over: the same 26.6 s of match video with skeletons. `claim-review.json` lists what the script claims, what it reads as tactics, and what it leaves out on purpose. `verification.json` checks frames, captions, voice timing and skeleton sync (within 0.017 s). |
| [`tiktok-coaching-editorial-20261103-v4/`](tiktok-coaching-editorial-20261103-v4/) | An earlier 31.3 s coaching reel, with the body points (`pose.json`) and findings (`analysis.json`) it was built from. |
| [`tactics-v3-hits-summary.json`](tactics-v3-hits-summary.json) | Counts from the tactics engine's hit table for the women's final: 336 hits, 6 with a shot name, 32 with a landing zone. |
| [`MANIFEST.json`](MANIFEST.json) | Every file in this repo with its size and hash. Check it with `python scripts\verify_manifest.py`. |

Local folder paths inside these records were replaced with `<workspace>` or `<home>`. Where that changed a file, `MANIFEST.json` also keeps the hash from before the change, which is the hash other records refer to.

These reels cannot be re-made from a fresh clone yet: they need the source match video and lab tools that are not in this repo. See [docs/reproducibility.md](../docs/reproducibility.md).
