# Models and licences

This repo has **no licence file**: the project's own code and records are all rights reserved. The parts below belong to others and keep their own licences. Check them before you reuse anything.

## In this repo

| File | What it is | Built from | Licence of the base |
|---|---|---|---|
| `fast/weights/rtmpose-s-ft1.onnx` | Main body-point model | RTMPose-s (body7, 256×192) by OpenMMLab, retrained for 8 rounds | Apache-2.0 |
| `fast/weights/rtmpose-t-ft3.onnx` | Smaller body-point model for far players | RTMPose-t by OpenMMLab, retrained for 8 + 12 rounds | Apache-2.0 |

**How they were retrained.** A bigger model (RTMPose-x, also OpenMMLab) marked the body points on frames from the 2026 YONEX Arise stream (CC BY 4.0). The small models learned to copy it. Training used other time windows of that stream only: sample clips 01–07 were kept out, and no China Open frames were used. The labels come from a model, not from people. Training records: [`train/ft1.json`](../train/ft1.json) for the main model (saved as `rtmpose-s-ft1.onnx`) and [`fast/runs/step2-training-provenance.json`](../fast/runs/step2-training-provenance.json) for the smaller one. The training code is in [`train/`](../train/).

## Downloaded by `scripts/get_models.py`

Not stored in this repo. Each download is checked against the hash recorded when the project was built.

| Group | Model | Used for | Source | Licence |
|---|---|---|---|---|
| `yolo` | YOLO26n and YOLO26s | Finding the players | Ultralytics; the script turns the official `.pt` files into ONNX on your computer | AGPL-3.0, or an Ultralytics Enterprise licence |
| `melo` | MeloTTS Chinese–English voice | The app's narration | sherpa-onnx release of MeloTTS | MIT |
| `kokoro` | Kokoro v1.1 multi-language voice | The technical commentary reel | sherpa-onnx release of Kokoro-82M | Apache-2.0 |
| `sensevoice` | SenseVoice small (int8) | Speech-to-text check of the narration | sherpa-onnx release of SenseVoice | Code MIT; weights under the FunASR model licence |
| `tracknet` | TrackNetV3 `TrackNet_best.pt` | Finding the shuttle | Official checkpoint link in [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) | MIT |
| `tracknet` | `badminton-pipeline-repro` at commit `1666675c` | The code that runs TrackNetV3 | [ychenfen/badminton-pipeline-repro](https://github.com/ychenfen/badminton-pipeline-repro); its own copy of the weights is not downloaded | MIT |

`yolo` and `melo` are all the app needs. Add `--studio` for the other three.

**About YOLO26.** Ultralytics releases these models under AGPL-3.0. If you share or offer a service built on this project while it uses them, AGPL terms apply to that use, unless you hold an Ultralytics Enterprise licence.

## Libraries and tools

| Part | Licence |
|---|---|
| OpenVINO, sherpa-onnx | Apache-2.0 |
| OpenCV | Apache-2.0 |
| NumPy, psutil, PyAV | BSD |
| Pillow | MIT-CMU |
| pytest | MIT |
| rtmlib (training, judge and comparison scripts; not in `requirements.txt`) | Apache-2.0 |
| ffmpeg (you supply it) | LGPL or GPL, depending on the build |
| Microsoft YaHei font (ships with Windows, not copied here) | Microsoft font licence |

Exact versions are in [`requirements.txt`](../requirements.txt).

## Match footage

None is stored here. Sources, licences and how to get each clip: [`sample-videos/README.md`](../sample-videos/README.md).

## Reproduction inputs and result videos

The full China Open source is ["Wang Zhiyi triumph in all-Chinese China Open finals I China Open 2025"](https://commons.wikimedia.org/wiki/File:Wang_Zhiyi_triumph_in_all-Chinese_China_Open_finals_I_China_Open_2025.webm), by **Badminton Sphere**, via Wikimedia Commons. Commons lists it as [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), permitting sharing and remixing with attribution. The release reels excerpt, reformat, slow or freeze parts of it and add model pose overlays, captions, and AI narration. No endorsement is implied. **Upstream broadcast rights have not been independently cleared.**

The source video's SHA256 is `2500471d4852a4a73e0f4eb25ad28171b3c54967f58e1cc7620af16b46ccbb52`. `scripts/get_footage.py --no-cut` fetches it anonymously and checks that hash.

The separate `reel-inputs-v1.zip` release bundle contains only 19 generated speech WAVs. Kokoro v1.1 generated the base and technical takes; MeloTTS generated the earlier coaching equivalents. See [`inputs-manifest.json`](../studio/reels/inputs-manifest.json) for each take's text, origin, and hash. These inputs are AI narration, not recordings of the players or a human coach.

The project's code licence does not grant rights to the source footage, models, font, or upstream broadcast. Each retains its own terms. No raw source-video corpus, model archive, or Microsoft font is uploaded in the reproduction bundle.
