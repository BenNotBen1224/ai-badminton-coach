import hashlib
import json
import pathlib
import re


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    current_path = HERE / "video-script.json"
    previous_path = ROOT / "badminton-coach" / "outputs" / "natural-three-moments-20261103-v7" / "script.json"
    current = load(current_path)
    previous = load(previous_path)
    rules = {rule["id"] for rule in load(HERE / "codebook.json")["rules"]}
    moments = {moment["id"]: moment for moment in current["moments"]}
    if len(moments) != len(current["moments"]):
        raise ValueError("Duplicate moment id.")
    if digest(pathlib.Path(current["source"])) != current["source_sha256"]:
        raise ValueError("Source video changed.")
    evidence = []
    for moment in current["moments"]:
        if not set(moment["rules"]).issubset(rules):
            raise ValueError(f"{moment['id']}: unknown codebook rule")
        report_path = HERE / moment["evidence"]
        report = load(report_path)
        if report["source_sha256"] != current["source_sha256"]:
            raise ValueError(f"{moment['id']}: different evidence source")
        for sample in report["samples"]:
            image = pathlib.Path(sample["overlay_image"])
            if digest(image) != sample["overlay_sha256"]:
                raise ValueError(f"Changed evidence image: {image}")
        evidence.append({
            "moment": moment["id"],
            "source_window_s": moment["source_window_s"],
            "observations_to_challenge": moment["observed_sequence"],
            "images": [sample["overlay_image"] for sample in report["samples"]],
            "sheets": [str(HERE / name) for name in moment.get("inspected_sheets", [])],
            "source_report": str(report_path),
            "source_report_sha256": digest(report_path),
        })
    ids = set()
    sentence_count = 0
    eligible_count = 0
    for beat in current["beats"]:
        if beat["id"] in ids:
            raise ValueError("Duplicate narration id.")
        ids.add(beat["id"])
        start, end = beat["source_support_s"]
        window_start, window_end = moments[beat["moment"]]["source_window_s"]
        if not window_start <= start < end <= window_end:
            raise ValueError(f"{beat['id']}: evidence outside selected moment")
        sentences = [s for s in re.split(r"[。！？!?]+", beat["text_zh"]) if s.strip()]
        if len(sentences) != 1:
            raise ValueError(f"{beat['id']}: expected one sentence per beat")
        sentence_count += len(sentences)
        if beat["why_next_eligible"]:
            if not beat.get("basis"):
                raise ValueError(f"{beat['id']}: no evidence basis")
            eligible_count += len(sentences)
    output = HERE / "script-review" / "input"
    output.mkdir(parents=True, exist_ok=True)
    packet = {
        "question": "Which script better teaches why a badminton shot is chosen from the opponent's current state and what must be handled next?",
        "candidates": [
            {"id": "A", "sentences": [
                {"id": f"A{i + 1:02d}", "text": beat["text_zh"]}
                for i, beat in enumerate(current["beats"])
            ]},
            {"id": "B", "sentences": [
                {"id": f"B{i + 1:02d}", "text": line["text"]}
                for i, line in enumerate(previous["lines"])
            ]},
        ],
        "comparison_limit": "The passages and script lengths differ; compare explanatory quality and unsupported claims, not duration or identical-footage preference.",
    }
    support = {
        "candidate": "A",
        "source": current["source"],
        "source_sha256": current["source_sha256"],
        "evidence": evidence,
        "sentence_moment_mapping": {
            f"A{i + 1:02d}": beat["moment"] for i, beat in enumerate(current["beats"])
        },
        "codebook": str(HERE / "codebook.json"),
        "codebook_verification": str(HERE / "codebook_verified.json"),
        "limits": current["limits"],
        "review_instruction": "Inspect the real images and challenge the supplied observations; they are assertions, not proof. Near/far changes at the end switch; shirt colors identify the narration's players.",
    }
    for name, value in (("scripts.json", packet), ("evidence.json", support)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "script_sha256": digest(current_path),
        "previous_script_sha256": digest(previous_path),
        "source_sha256": current["source_sha256"],
        "sentences": sentence_count,
        "author_proposed_why_next": eligible_count,
        "author_proposed_fraction": eligible_count / sentence_count,
        "independent_rating": "pending",
        "rendered_evidence_and_audio": "not_yet_produced",
        "mapping": {"A": str(current_path), "B": str(previous_path)},
        "input_hashes": {name: digest(output / name) for name in ("scripts.json", "evidence.json")},
    }
    (HERE / "script-preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("sentences", "author_proposed_why_next", "independent_rating")}))


if __name__ == "__main__":
    main()
