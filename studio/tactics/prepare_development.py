import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / "labels" / "labels.json"
DEVELOPMENT_IDS = {"R1", "R2"}


def main():
    raw = SOURCE.read_bytes()
    document = json.loads(raw)
    development = [r for r in document["rallies"] if r["id"] in DEVELOPMENT_IDS]
    if {r["id"] for r in development} != DEVELOPMENT_IDS:
        raise ValueError("The fixed development partition is incomplete.")
    output = ROOT / "development"
    output.mkdir(exist_ok=True)
    result = {
        "labeler": document["labeler"],
        "method": document["method"],
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "purpose": "Development-only visual annotations for reproducing contact-phase errors. No player/time-specific rules may be added to the engine.",
        "limits": "AI-authored annotations, not human-certified contacts. Preserve individual low-confidence fields and verify ambiguous examples against the source.",
        "rallies": development,
    }
    (output / "labels.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    partition = {
        "development": sorted(DEVELOPMENT_IDS),
        "evaluation": [r["id"] for r in document["rallies"] if r["id"] not in DEVELOPMENT_IDS],
        "source_sha256": result["source_sha256"],
        "baseline_versions": ["tactics-engine-20261104", "tactics-engine-20261105-v2"],
        "limitation": "Aggregate scores on all five rallies have already been observed. The remaining three are not a pristine blind generalization sample, although their exact labels remain unavailable to the engine author.",
    }
    (output / "partition.json").write_text(json.dumps(partition, indent=2), encoding="utf-8")
    print(json.dumps({"rallies": len(development), "contacts": sum(len(r["hits"]) for r in development),
                      "output": str(output / "labels.json")}))


if __name__ == "__main__":
    main()
