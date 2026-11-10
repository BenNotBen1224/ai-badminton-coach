import collections
import json
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parent


def build():
    audit = json.loads((ROOT / "books" / "audit.json").read_text(encoding="utf-8"))
    if audit["errors"]:
        raise ValueError("Source audit has unresolved errors; do not publish the codebook.")
    book = json.loads((ROOT / "codebook.json").read_text(encoding="utf-8"))
    passages = {
        row["passage_id"]: row
        for row in map(json.loads, (ROOT / "books" / "verified_passages.jsonl").read_text(encoding="utf-8").splitlines())
    }
    errors = []
    seen = set()
    rendered = {}
    for rule in book["rules"]:
        if rule["id"] in seen:
            errors.append(f"duplicate rule: {rule['id']}")
        seen.add(rule["id"])
        for key in ("opponent_state", "choice", "expected_reply", "check", "required_evidence", "do_not_apply"):
            if not rule.get(key):
                errors.append(f"{rule['id']}: missing {key}")
        citations = [passages[p] for p in rule["sources"] if p in passages]
        missing = set(rule["sources"]) - set(passages)
        if missing:
            errors.append(f"{rule['id']}: missing passages {sorted(missing)}")
        if len({p["canonical_book"] for p in citations}) < 2:
            errors.append(f"{rule['id']}: fewer than two distinct source works")
        for citation in citations:
            rendered[citation["passage_id"]] = citation
    for passage_id in book["interpretation_guard"]["sources"]:
        if passage_id not in passages:
            errors.append(f"missing interpretation source: {passage_id}")
        else:
            rendered[passage_id] = passages[passage_id]
    quote_words = collections.Counter()
    quote_cjk = collections.Counter()
    for p in rendered.values():
        quote_words[p["canonical_book"]] += len(p["quote"].split())
        quote_cjk[p["canonical_book"]] += len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", p["quote"]))
    for source, words in quote_words.items():
        if words > 200:
            errors.append(f"{source}: {words} quoted words exceeds this deliverable's 200-word cap")
    for source, characters in quote_cjk.items():
        if characters > 400:
            errors.append(f"{source}: {characters} quoted CJK characters exceeds this deliverable's 400-character cap")
    output = {
        **book,
        "citations": [
            {key: p[key] for key in ("passage_id", "book", "canonical_book", "title", "locator", "quote", "source_url", "text_sha256_norm")}
            for p in rendered.values()
        ],
        "verification": {
            "rules": len(book["rules"]),
            "distinct_source_works": len({p["canonical_book"] for p in rendered.values()}),
            "citations": len(rendered),
            "quoted_whitespace_tokens_by_work": dict(quote_words),
            "quoted_cjk_characters_by_work": dict(quote_cjk),
            "errors": errors,
            "scope": "Source identity, verbatim quotations and minimum distinct-work counts are checked. Source agreement is not a measured probability or independent experimental validation.",
        },
    }
    (ROOT / "codebook_verified.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["verification"], ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(build())
