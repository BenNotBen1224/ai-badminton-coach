import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[3]
OUT = RUN_DIR / "render"
SCRIPT_PATH = RUN_DIR / "render-script.json"
HELPER = ROOT / "obj2" / "work" / "TACTICS" / "original-render" / "render.py"
V2 = ROOT / "badminton-coach" / "outputs" / "tactics-original-editorial-20261105" / "tiktok-viral-v2"
RANGES = {
    "space": [57.8, 64.8],
    "time": [126.6, 134.0],
    "options": [272.8, 285.0],
}
SPOKEN = {
    "space": "韩悦调王祉怡，抓的是回位空当。前后拉开，别等她站稳了再打。",
    "time": "王祉怡被动，先过渡拆连贯。硬顶快球，容易把韩悦的下一拍喂舒服。",
    "options": "换韩悦被压，王祉怡再往前抢，就打她身后，逼她掉头。防守想翻身，先让对手打不顺。",
}

os.environ["TACTICS_EDITORIAL_OUT"] = str(OUT)
os.environ["TACTICS_EDITORIAL_SCRIPT"] = str(SCRIPT_PATH)
os.environ["TACTICS_EDITORIAL_FORMAT"] = "vertical_tiktok"

spec = importlib.util.spec_from_file_location("editorial_renderer", HELPER)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def validate_script(script):
    source = script["source_time_mapping"]
    if source["source_sha256"] != renderer.sha(renderer.SOURCE):
        raise ValueError("Script source hash does not match the original reel")
    moments = script["moments"]
    beats = script["beats"]
    if [m["id"] for m in moments] != list(RANGES):
        raise ValueError("The three source passages must remain in their original order")
    if [b["id"] for b in beats] != list(RANGES):
        raise ValueError("Exactly one narrated beat is required per source passage")
    for moment, beat in zip(moments, beats):
        if moment["range"] != RANGES[moment["id"]] or beat["range"] != RANGES[beat["id"]]:
            raise ValueError(f"Source range changed: {moment['id']}")
        if beat["text"] != SPOKEN[beat["id"]]:
            raise ValueError(f"Spoken wording changed after editorial direction: {beat['id']}")
        if "".join(beat["caption_chunks"]) != beat["text"] or len(beat["caption_chunks"]) < 2:
            raise ValueError(f"Captions must reproduce speech in phrase-sized chunks: {beat['id']}")
        if beat["hold"] != 0 or beat.get("target_speed", 1.0) != 1.0:
            raise ValueError(f"No freezes or planned slow motion: {beat['id']}")
    if not beats[0].get("hook") or beats[0].get("hook_text") != script["title_zh"]:
        raise ValueError("The opening title must use the approved question without separate narration")
    if script["identity_review"]["mapping"] != {
        "red_kit": "Wang Zhi Yi",
        "white_kit": "Han Yue",
    }:
        raise ValueError("Player identity mapping is not the reviewed source mapping")
    analysis = script["coach_analysis"]
    if [b["id"] for b in analysis["beats"]] != list(RANGES):
        raise ValueError("Every narration beat needs a matching observation and coaching analysis")
    for item in analysis["beats"]:
        if not item["observed"] or not item["tactical_read"] or not item["limitations"]:
            raise ValueError(f"Incomplete evidence boundaries: {item['id']}")


def add_reused_context_evidence():
    source = V2 / "context-evidence.json"
    evidence = renderer.read(source)
    if not evidence:
        raise ValueError("The prior source/pose context evidence is empty")
    for item in evidence:
        if item["moment"] not in RANGES or not RANGES[item["moment"]][0] <= item["source_time_s"] <= RANGES[item["moment"]][1]:
            raise ValueError(f"Context evidence falls outside the unchanged passage: {item['id']}")
        if renderer.sha(Path(item["image"])) != item["image_sha256"]:
            raise ValueError(f"Source picture changed: {item['id']}")
        if renderer.sha(Path(item["pose_cache"])) != item["pose_cache_sha256"]:
            raise ValueError(f"Pose cache changed: {item['id']}")
        if item["pose_sync_error_s"] > 1 / 60 + .0002:
            raise ValueError(f"Source/pose sync exceeds one frame: {item['id']}")
    renderer.write(OUT / "context-evidence.json", evidence)
    inputs = renderer.read(OUT / "inputs.json")
    inputs["inputs"][str(Path(__file__).resolve())] = renderer.sha(Path(__file__))
    inputs["inputs"][str(source.resolve())] = renderer.sha(source)
    renderer.write(OUT / "inputs.json", inputs)


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing render folder: {OUT}")
    script = renderer.read(SCRIPT_PATH)
    validate_script(script)
    renderer.stage_evidence_and_inputs(script, create_frozen=True)
    add_reused_context_evidence()
    voice = renderer.synth(script)
    plan = renderer.make_schedule(script, voice)
    if any(beat["moving_speed"] < .95 for beat in plan["beats"]):
        raise ValueError("Shorten speech before materially slowing the source passages")
    renderer.write(OUT / "planned-schedule.json", plan)
    for beat in plan["beats"]:
        print(
            beat["id"],
            "display", beat["display_span_s"],
            "speech", beat["speech_span_s"],
            "speed", round(beat["moving_speed"], 3),
            flush=True,
        )
    print("PLANNED", plan["duration_s"], "seconds;", plan["frame_count"], "frames", flush=True)


def verify():
    script = renderer.read(OUT / "script.json")
    validate_script(script)
    renderer.validate_reviewed_script = validate_script
    renderer.verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "encode", "verify"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
    elif stage == "encode":
        renderer.encode()
    else:
        verify()


if __name__ == "__main__":
    main()
