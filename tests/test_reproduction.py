import json
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "studio" / "tactics" / "score_v3.py"
INPUTS = SCORE.parent / "inputs"


def rescore(out, *args):
    return subprocess.run([sys.executable, str(SCORE), "--out", str(out), *map(str, args)],
                          capture_output=True, text=True)


def test_frozen_tactics_matches_every_historical_metric(tmp_path):
    result = rescore(tmp_path, "--check-reference")
    assert result.returncode == 0, result.stderr
    actual = json.loads((tmp_path / "summary.json").read_text())
    emitted = actual["variants"]["as_emitted"]
    assert emitted["totals"]["matched"] == 32
    assert emitted["totals"]["label"] == 96
    assert emitted["totals"]["engine"] == 91
    assert emitted["recall"] == 1 / 3
    assert emitted["all_gates_passed"] is False


def test_changed_predictions_change_scores(tmp_path):
    document = json.loads((INPUTS / "hits.json").read_text())
    for hit in document["hits"]:
        hit["time_s"] += 10000
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(document))
    assert rescore(tmp_path / "out", "--hits", changed).returncode == 0
    actual = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert actual["variants"]["as_emitted"]["recall"] == 0
    assert rescore(tmp_path / "ref", "--hits", changed, "--check-reference").returncode != 0


def test_changed_labels_change_scores(tmp_path):
    document = json.loads((INPUTS / "labels.json").read_text())
    for rally in document["rallies"]:
        for hit in rally["hits"]:
            hit["hitter"] = "unmatched"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(document))
    assert rescore(tmp_path / "out", "--labels", changed).returncode == 0
    actual = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert actual["variants"]["as_emitted"]["recall"] == 0


def test_missing_tactics_input_fails(tmp_path):
    result = rescore(tmp_path / "out", "--labels", tmp_path / "missing.json")
    assert result.returncode != 0
    assert not (tmp_path / "out" / "summary.json").exists()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_or_changed_voice_is_not_accepted(tmp_path):
    renderer = load_script("render_reels")
    manifest = {"editions": {"technical": [{"file": "voice.wav", "sha256": hashlib.sha256(b"right").hexdigest()}]}}
    with pytest.raises(ValueError, match="Missing or changed narration"):
        renderer.checked_voices(tmp_path, "technical", manifest)
    (tmp_path / "voice.wav").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Missing or changed narration"):
        renderer.checked_voices(tmp_path, "technical", manifest)
    (tmp_path / "voice.wav").write_bytes(b"right")
    assert len(renderer.checked_voices(tmp_path, "technical", manifest)) == 1


def test_wrong_source_is_rejected_before_render(tmp_path):
    source = tmp_path / "wrong.webm"
    source.write_bytes(b"not the source")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_reels.py"),
                             "--edition", "technical", "--source", str(source),
                             "--inputs", str(tmp_path), "--out", str(tmp_path / "out")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Source SHA256 differs" in result.stderr
    assert not (tmp_path / "out").exists()


def test_bundle_member_hashes_and_exact_list_are_checked(tmp_path):
    fetcher = load_script("get_reel_inputs")
    archive = tmp_path / "inputs.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("voices/test.wav", b"right")
    manifest = {"archive": {"sha256": fetcher.sha(archive)},
                "editions": {"test": [{"file": "voices/test.wav", "bytes": 5,
                                       "sha256": hashlib.sha256(b"right").hexdigest()}]}}
    assert fetcher.unpack(archive, tmp_path / "out", manifest) == 1
    assert (tmp_path / "out" / "voices" / "test.wav").read_bytes() == b"right"
    manifest["editions"]["test"][0]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValueError, match="Changed archive member"):
        fetcher.unpack(archive, tmp_path / "bad", manifest)
    assert not (tmp_path / "bad").exists()
    manifest["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive SHA256"):
        fetcher.unpack(archive, tmp_path / "bad", manifest)


def test_latest_wording_and_source_mapping_stay_fixed():
    reels = ROOT / "studio" / "reels"
    script = json.loads((reels / "technical-script.json").read_text(encoding="utf-8"))
    assert script["hook"]["lines"] == ["韩悦对上王祉怡", "这几拍，攻防节奏拉满！"]
    lines = {line["id"]: line["show"] for beat in script["beats"] for line in beat["lines"]}
    assert lines["space-3"] == "神仙防守！"
    assert lines["time-2"] == "韩悦底线暴力跳杀！"
    assert lines["options-2"] == "韩悦接杀走对角，把王祉怡拉开！"
    assert lines["options-4"] == "韩悦下压抢网，连贯拉满！"
    mapping = json.loads((reels / "base-frame-map.json").read_text(encoding="utf-8"))
    assert [row["output_frame"] for row in mapping["frames"]] == list(range(798))
    plan = json.loads((reels / "base-plan.json").read_text(encoding="utf-8"))
    assert [beat["range"] for beat in plan["beats"]] == [[57.8, 64.8], [126.6, 134.0], [272.8, 285.0]]
    for row in mapping["frames"]:
        assert abs(row["pose_source_time_s"] - row["source_position"] / 60) < 0.017
