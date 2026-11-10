"""Learn which coaching points badminton videos stress, from open transcripts.

Usage: python corpus_learn.py <corpus_dir>
Reads ytc_text*.jsonl, vidchapters_hits.jsonl, yodas2_*.jsonl from <corpus_dir>,
writes transcripts.jsonl, counts.tsv, learned.json there.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

STRONG = re.compile(
    r"badminton|bádminton|badmington|羽毛球|羽球|bulu ?tangkis|バドミントン|배드민턴|แบดมินตัน|cầu lông|"
    r"बैडमिंटन|бадминтон|shuttlecock|shuttle cock", re.I)
DOMAIN = re.compile(
    r"\b(shuttle\w*|birdie|smash\w*|drop ?shots?|net ?shots?|net kill|net play|clears?|lifts?|rackets?|racquets?|"
    r"footwork|forehand|backhand|serves?|serving|rally|rallies|lunge\w*|split[\s-]?step|grip|drives?|doubles|"
    r"singles|court|baseline|forecourt|rearcourt|rear court|front court|mid ?court)\b|"
    r"杀球|吊球|挑球|步法|网前|发球|高远球|平抽|反手|正手|球拍", re.I)
WORD = re.compile(r"[A-Za-z']+|[\u4e00-\u9fff]")
# Same badminton-only names, events, racket lines and shot names as ytc_scan_ext.py.
TITLE_EXTRA = re.compile(
    r"lee chong wei|\blin dan\b|chen long|axelsen|momota|tai tzu.?ying|carolina mar[ií]n|\bp\.?\s?v\.? sindhu|"
    r"saina nehwal|kidambi|lakshya sen|h\.?\s?s\.? prannoy|satwiksairaj|chirag shetty|kevin sanjaya|"
    r"marcus (fernaldi )?gideon|hendra setiawan|lee zii jia|anthony (sinisuka )?ginting|jonatan christie|"
    r"shi yu.?qi|chou tien.?chen|ratchanok|akane yamaguchi|nozomi okuhara|an se.?young|chen yu.?fei|"
    r"he bing.?jiao|lee yong.?dae|yoo yeon.?seong|fu hai.?feng|zheng si.?wei|huang ya.?qiong|taufik hidayat|"
    r"peter gade|kunlavut|loh kean yew|anders antonsen|wang zhi.?yi|greysia polii|gopichand|"
    r"\bbwf\b|thomas cup|uber cup|sudirman cup|all england open|shuttle time|"
    r"astrox|nanoflare|arcsaber|voltric|nanoray|duora|thruster|jetspeed|auraspeed|axforce|aeronaut|"
    r"jump smash|net kill|backhand clear|forehand clear|high clear|tumbling net|spinning net|"
    r"林丹|李宗伟|谌龙|安赛龙|桃田|戴资颖|石宇奇|陈雨菲|郑思维|黄雅琼|国羽|杀球|吊球|高远球|网前小球", re.I)
NOT_BADMINTON = re.compile(r"no-eh|jianzi|kick(ing)? shuttle|shuttlecock kick|đá cầu", re.I)
OTHER_SPORT = re.compile(r"table tennis|ping[\s-]?pong|\bgolf\w*|pickleball|\bsquash\b|\btennis\b|volleyball|"
                         r"basketball|football|soccer|cricket|baseball|hockey", re.I)

TITLE_KIND = [
    ("review", re.compile(r"\breview|unbox|string(ing)?\b|racket test|racquet test|shoes?\b|\bbuy(ing)?\b|"
                          r"choos(e|ing) (a |the |your )?(badminton )?(racket|racquet|shoes?|strings?)", re.I)),
    ("coaching", re.compile(r"how to|tutorial|\btips?\b|technique|lesson|drills?|training|coach|footwork|beginner|"
                            r"learn|mistakes?|improve|教学|技术|步法", re.I)),
    ("match", re.compile(r"\bvs\.?\b|\bv\b|final|highlights?|championships?|\bopen\b|\bcup\b|\bgames\b|"
                         r"tournament|\bmatch\b|\bmd\b|\bws\b|\bms\b|\bwd\b|\bxd\b|比赛|决赛", re.I)),
]
RATE_KIND = {
    "coaching": re.compile(r"\b(you|your|should|try|practi[cs]e|drills?|technique|tips?|how to|make sure|"
                           r"beginners?|lesson|learn|exercise)\b", re.I),
    "match": re.compile(r"\b(game point|match point|championships?|final|semi[\s-]?final|quarter[\s-]?final|"
                        r"leads?|leading|interval|umpire|challenge|seeded|world number|tournament|"
                        r"eleven|twenty[\s-]?one)\b", re.I),
    "review": re.compile(r"\b(review|strings?|tension|grams?|price|unbox\w*|balance point|stiff|flex|shaft|"
                         r"head[\s-]?heavy|head[\s-]?light)\b", re.I),
}

# key, plain English label, Chinese label, pattern, what our pose pipeline measures for it (None = cannot)
CONCEPTS = [
    ("split_step", "Split step as opponent hits", "对手击球时启动垫步",
     r"split[\s-]?step|split[\s-]?jump|\bpre[\s-]?jump|启动步|垫步|分腿跳", "footwork.split_step"),
    ("ready_position", "Ready position", "准备姿势",
     r"ready (position|stance|pose|posture|for (the )?next)|(be|get|stay|always|so|are) ready\b|"
     r"base (position|stance)|athletic (stance|position)|准备姿势|准备动作", "technique.stance_width"),
    ("racket_up", "Racket up, prepare early", "早举拍早准备",
     r"(racket|racquet)( head)? up|raise (your |the )?(racket|racquet)|(racket|racquet) preparation|"
     r"prepar\w* (your |the )?(racket|racquet)|prepare early|early preparation|backswing|back swing|举拍|引拍", None),
    ("first_step", "Fast reaction, first step", "反应快第一步快",
     r"first step|\breact\w*|anticipat\w*|read(ing)? (the|your) (opponent|shot)|quick start|反应|预判|第一步",
     "footwork.first_step"),
    ("footwork", "Footwork pattern", "步法",
     r"footwork|foot work|chass[eé]|cross[\s-]?over step|cross[\s-]?step|shuffle|步法|并步|交叉步",
     "footwork.step_pattern"),
    ("lunge", "Lunge", "弓步上网", r"\blung(e|es|ed|ing)\b|弓步|跨步", "technique.lunge_depth"),
    ("recovery", "Recover to the centre", "击球后回中",
     r"\brecover\w*|back to (the )?(base|cent(er|re)|middle)|return(ing)? to (the )?(base|cent(er|re)|middle)|"
     r"come back to (the )?(base|cent(er|re)|middle)|回中|回位", "footwork.recovery"),
    ("contact_point", "Hit high and in front", "高点、身前击球",
     r"contact point|point of contact|hitting point|highest point|high point|in front of (you|your body|the body)|"
     r"hit (it|the shuttle|the bird) (early|high|in front)|take (it|the shuttle) early|"
     r"(hit|meet|take) (it |the shuttle |the bird )?at the (top|highest)|击球点|最高点",
     "contact.reach_height"),
    ("arm_extension", "Straight arm at contact", "击球手臂伸展",
     r"(straight|extended|full) arm|arm (straight|extended|fully extended)|extend\w* (your )?(arm|hand)|"
     r"straighten (your |the )?(arm|elbow)|high elbow|elbow (up|high|goes up|leads?)|lead with (the |your )?elbow|"
     r"手臂伸直|伸直手臂", "contact.arm_extension"),
    ("grip", "Grip", "握拍", r"\bgrips?\b|gripping|thumb|panhandle|握拍", None),
    ("forearm_wrist", "Forearm rotation, wrist", "前臂旋转与手腕",
     r"forearm|pronat\w*|supinat\w*|\bwrists?\b|finger power|手腕|前臂|内旋", None),
    ("body_rotation", "Turn body, shift weight", "转体与重心转移",
     r"rotat\w* (your |the )?(body|hips?|shoulders?|trunk|torso)|body rotation|hip rotation|shoulder rotation|"
     r"weight transfer|transfer (your |the )?weight|turn (your |the )?(body|hips?|shoulders?)|side[\s-]?on|"
     r"sideways|转体|转髋|重心转移|侧身", None),
    ("non_racket_arm", "Non-racket arm for balance", "非持拍手平衡",
     r"non[\s-]?(racket|racquet|hitting) (arm|hand)|(left|other|free|opposite) (arm|hand)|左手", None),
    ("knee_bend", "Bend knees, stay low", "屈膝降重心",
     r"bend (your )?knees?|knees? (bent|bend)|stay low|get low|low (cent(er|re) of gravity|stance|position)|"
     r"lower (your )?(body|cent(er|re) of gravity)|屈膝|降低重心|重心低", "technique.knee_bend"),
    ("stance_width", "Feet width", "两脚站距",
     r"(wide|wider|narrow) stance|shoulder[\s-]?width|feet (apart|wide|wider)|stance width|站距|分腿站位|支撑宽度",
     "technique.stance_width"),
    ("relax_power", "Relax, then power", "放松再发力", r"\brelax\w*|\bloose\b|\btense\b|发力|放松", None),
    ("watch_shuttle", "Watch the shuttle", "盯球",
     r"(watch|eyes? on|look at|keep (your )?eyes? on) the (shuttle\w*|bird\w*)|盯球", None),
    ("timing", "Timing", "击球时机", r"\btiming\b|too late|too early|时机", None),
    ("deception", "Deception", "假动作", r"decept\w*|disguis\w*|\bfake\w*|假动作", None),
    ("placement", "Placement, tactics", "落点与战术",
     r"placement|shot selection|tactic\w*|strateg\w*|open space|\bcorners?\b|落点|线路|战术", "tactics.initiative"),
    ("consistency", "Consistency, fewer errors", "稳定少失误",
     r"consisten\w*|unforced error|\berrors?\b|失误|稳定", None),
    ("fitness", "Fitness, warm-up", "体能热身",
     r"fitness|stamina|endurance|stretch\w*|warm[\s-]?up|conditioning|体能|热身|拉伸", None),
    ("mental", "Mental focus", "心态",
     r"\bmental\w*|confiden\w*|nervous|concentrat\w*|\bcalm\b|心态|紧张|专注", None),
    ("follow_through", "Follow-through", "随挥", r"follow[\s-]?through|随挥|收拍", None),
    ("equipment", "Equipment", "器材",
     r"\bstrings?\b|stringing|tension|\bgrams?\b|\bshoes?\b|balance point|head[\s-]?heavy|head[\s-]?light|球线|球鞋",
     None),
]
CONCEPT_RE = {key: re.compile(pat, re.I) for key, _, _, pat, _ in CONCEPTS}
STRESS = re.compile(
    r"\b(most important|important|importance|the key|key (thing|point|is)|crucial|essential|vital|critical|"
    r"fundamental|make sure|remember|focus on|concentrate on|the secret|common mistake|biggest mistake|"
    r"don't forget|do not forget|pay attention)\b|关键|重要|一定要|记住|核心", re.I)
DOWNPLAY = re.compile(
    r"(doesn't|does not|don't|do not|won't) (really )?matter|not (that |so |very |too )?important|"
    r"isn't (that |so |very )?important|less important|don't worry|do not worry|no need to|not necessary|"
    r"unnecessary|you don't (have|need) to|you do not (have|need) to|not a big deal|overrated|"
    r"不重要|没必要|无所谓|不需要", re.I)
MOST = re.compile(r"\b(the )?(most important|key|main|biggest|number one) (thing|point|part|tip|mistake|key)s?\b"
                  r"( (is|are|to remember( is)?|here is|i see|that))?", re.I)
HEDGE_ZH = re.compile(r"不是|不能证明|不照抄|估计|投影|不代表|仅供")
WINDOW = 110


def words(text):
    return len(WORD.findall(text))


def lang_of(text):
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = re.findall(r"[A-Za-z']+", text.lower())
    if cjk > 0.3 * max(1, cjk + len(latin)):
        return "zh"
    common = sum(w in {"the", "and", "you", "to", "is", "it", "that", "a", "of", "in"} for w in latin)
    return "en" if latin and common / len(latin) > 0.12 else "other"


def load(corpus):
    docs = []
    for p in sorted(corpus.glob("ytc_text*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            docs.append({"source": "youtube-commons", "video_id": r["video_id"], "title": r["title"],
                         "channel": r["channel"], "text": r["text"] or "", "licence": r.get("license") or "CC-BY"})
    p = corpus / "vidchapters_hits.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines() if p.exists() else []:
        r = json.loads(line)
        segs = [s for s in (r["asr"] or {}).get("text") or [] if isinstance(s, str)]
        docs.append({"source": "vidchapters", "video_id": r["video_id"], "title": r["title"], "channel": None,
                     "text": " ".join(s.strip() for s in segs), "licence": "VidChapters research use"})
    for p in sorted(corpus.glob("yodas2_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            text = " ".join(r["utts"][k] for k in sorted(r["utts"]))
            docs.append({"source": "yodas2", "video_id": r["video_id"], "title": None, "channel": None,
                         "text": text, "licence": "CC (YODAS2)"})
    best = {}
    for d in docs:
        d["words"] = words(d["text"])
        if d["video_id"] not in best or d["words"] > best[d["video_id"]]["words"]:
            best[d["video_id"]] = d
    return docs, list(best.values())


def relevance(d):
    title = d["title"] or ""
    strong = len(STRONG.findall(d["text"]))
    other = len(OTHER_SPORT.findall(d["text"]))
    domain = len(DOMAIN.findall(d["text"]))
    per_k = 1000 * domain / max(1, d["words"])
    titled = bool(STRONG.search(title) or TITLE_EXTRA.search(title)) and not NOT_BADMINTON.search(title)
    d.update(strong=strong, other_sport=other, domain=domain, domain_per_k=round(per_k, 1), titled=titled)
    if NOT_BADMINTON.search(title) or NOT_BADMINTON.search(d.get("channel") or ""):
        return False
    if titled:
        return d["words"] >= 60
    return d["words"] >= 80 and strong >= 2 and strong >= other and domain >= 6 and per_k >= 4


TOPICS = {
    "footwork": r"footwork|foot ?work|\bsteps?\b|movement|\bmov(e|ing)\b|split[\s-]?step|lunge|chass[eé]|步法|启动",
    "overhead": r"smash|\bclears?\b|drop ?shot|overhead|jump|杀球|高远球|吊球",
    "net": r"\bnet\b|flick|net ?kill|tumbl|spinning|网前|搓球|推球",
    "defence": r"defen[cs]e|\bblock|\blift|smash return|receiv|接杀|防守",
    "serve": r"\bserv(e|es|ing)\b|发球",
    "grip": r"\bgrips?\b|握拍",
    "tactics": r"tactic|strateg|singles|doubles|placement|战术",
}
TOPIC_RE = {k: re.compile(p, re.I) for k, p in TOPICS.items()}


def topics_of(d):
    if d["title"]:
        return [k for k, rx in TOPIC_RE.items() if rx.search(d["title"])]
    head = " ".join(d["text"].split()[:400])
    return [k for k, rx in TOPIC_RE.items() if len(rx.findall(head)) >= 3]


OFF_TOPIC_TITLE = re.compile(r"vlog|origami|hanetsuki|giveaway|\bbags?\b|overview|diagonal|dimension|measurement|"
                             r"\bnews\b|interview|ranked|seeks|attempt|calluses|practical\b|\bPE\b|diet|다이어트", re.I)


def kind_of(d):
    if d["title"] and OFF_TOPIC_TITLE.search(d["title"]):
        return "other"
    for kind, pat in TITLE_KIND:
        if d["title"] and pat.search(d["title"]):
            return kind
    rates = {k: 1000 * len(p.findall(d["text"])) / max(1, d["words"]) for k, p in RATE_KIND.items()}
    if rates["review"] > 6:
        return "review"
    if rates["match"] > 3:
        return "match"
    if rates["coaching"] > 35:
        return "coaching"
    return "other"


def snippet(text, a, b):
    s = text[max(0, a - 60):b + 60]
    return re.sub(r"\s+", " ", s).strip()


def analyse(docs):
    stats = {k: Counter() for k, *_ in CONCEPTS}
    wstats = {k: Counter() for k, *_ in CONCEPTS}
    per_channel = Counter(d.get("channel") or d["video_id"] for d in docs)
    quotes, most_hits, most_examples, downplay_quotes = defaultdict(list), Counter(), [], defaultdict(list)
    total_words = sum(d["words"] for d in docs)
    for d in docs:
        text = d["text"]
        weight = 1 / per_channel[d.get("channel") or d["video_id"]]
        for key, rx in CONCEPT_RE.items():
            found = stressed = downplayed = False
            for m in rx.finditer(text):
                found = True
                stats[key]["mentions"] += 1
                ctx = text[max(0, m.start() - WINDOW):m.end() + WINDOW]
                if STRESS.search(ctx):
                    stressed = True
                    stats[key]["stress_mentions"] += 1
                    if len(quotes[key]) < 40:
                        quotes[key].append((d["video_id"], snippet(text, m.start(), m.end())))
                if DOWNPLAY.search(ctx):
                    downplayed = True
                    if len(downplay_quotes[key]) < 40:
                        downplay_quotes[key].append((d["video_id"], snippet(text, m.start(), m.end())))
            stats[key]["docs"] += found
            stats[key]["stressed_docs"] += stressed
            stats[key]["downplayed_docs"] += downplayed
            wstats[key]["docs"] += weight * found
            wstats[key]["stressed_docs"] += weight * stressed
        for m in MOST.finditer(text):
            tail = text[m.end():m.end() + 140]
            hit = [k for k, rx in CONCEPT_RE.items() if rx.search(tail)]
            for k in set(hit):
                most_hits[k] += 1
            if hit and len(most_examples) < 400:
                most_examples.append((d["video_id"], re.sub(r"\s+", " ", text[m.start():m.end() + 140]).strip()))
    n = max(1, len(docs))
    channels = max(1, len(per_channel))
    rows = []
    for key, en, zh, _, detector in CONCEPTS:
        s = stats[key]
        rows.append({
            "key": key, "label": en, "label_zh": zh, "detector": detector,
            "mention_share": round(s["docs"] / n, 3), "stressed_share": round(s["stressed_docs"] / n, 3),
            "channel_mention_share": round(wstats[key]["docs"] / channels, 3),
            "channel_stressed_share": round(wstats[key]["stressed_docs"] / channels, 3),
            "downplayed_share": round(s["downplayed_docs"] / n, 3),
            "per_10k_words": round(1e4 * s["mentions"] / max(1, total_words), 2),
            "most_important_hits": most_hits[key], "docs": s["docs"], "stressed_docs": s["stressed_docs"],
            "downplayed_docs": s["downplayed_docs"],
            "quotes": quotes[key][:6], "downplay_quotes": downplay_quotes[key][:6],
        })
    rows.sort(key=lambda r: (-r["channel_stressed_share"], -r["stressed_share"], -r["mention_share"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows, most_examples


def score_narration(path, rows):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip() and not re.match(r"^[\d.]+[–-]", l)]
    rank = {r["key"]: r["rank"] for r in rows}
    hits = sorted({k for k, rx in CONCEPT_RE.items() if rx.search(text)}, key=lambda k: rank[k])
    return {"lines": len(lines), "concepts": [{"key": k, "rank": rank[k]} for k in hits],
            "hedge_lines": sum(bool(HEDGE_ZH.search(l)) for l in lines)}


def main(corpus_dir, narration=None):
    corpus = Path(corpus_dir)
    raw, docs = load(corpus)
    counts = defaultdict(Counter)
    for d in raw:
        counts[d["source"]]["fetched"] += 1
    for d in docs:
        d["lang"] = lang_of(d["text"])
        d["relevant"] = relevance(d)
        d["kind"] = kind_of(d) if d["relevant"] else None
        counts[d["source"]]["unique"] += 1
        if d["relevant"]:
            counts[d["source"]]["relevant"] += 1
            counts[d["source"]][f"lang_{d['lang']}"] += 1
            counts[d["source"]][f"kind_{d['kind']}"] += 1
    with (corpus / "transcripts.jsonl").open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    cols = sorted({c for v in counts.values() for c in v})
    lines = ["source\t" + "\t".join(cols)] + [s + "\t" + "\t".join(str(counts[s][c]) for c in cols) for s in counts]
    total = Counter()
    for v in counts.values():
        total.update(v)
    lines.append("total\t" + "\t".join(str(total[c]) for c in cols))
    (corpus / "counts.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    usable = [d for d in docs if d["relevant"] and d["lang"] in ("en", "zh")]
    coaching = [d for d in usable if d["kind"] == "coaching"]
    match = [d for d in usable if d["kind"] == "match"]
    for d in coaching:
        d["topics"] = topics_of(d)
    coach_rows, most_examples = analyse(coaching)
    all_rows, _ = analyse(usable)
    match_rows, _ = analyse(match) if match else ([], [])
    by_topic = {}
    for topic in TOPICS:
        subset = [d for d in coaching if topic in d["topics"]]
        if subset:
            rows, _ = analyse(subset)
            by_topic[topic] = {"videos": len(subset),
                               "channels": len({d.get("channel") or d["video_id"] for d in subset}),
                               "rows": rows}
    result = {
        "corpus": {"relevant_videos": sum(d["relevant"] for d in docs), "analysed_videos": len(usable),
                   "coaching_videos": len(coaching), "match_videos": len(match),
                   "coaching_channels": len({d.get("channel") or d["video_id"] for d in coaching}),
                   "coaching_words": sum(d["words"] for d in coaching),
                   "analysed_words": sum(d["words"] for d in usable)},
        "coaching": coach_rows, "all": all_rows, "match": match_rows, "by_topic": by_topic,
        "coaching_videos": [{"video_id": d["video_id"], "source": d["source"], "title": d["title"],
                             "channel": d.get("channel"), "words": d["words"], "topics": d["topics"]}
                            for d in coaching],
        "most_important_examples": most_examples[:80],
        "old_narration": score_narration(Path(narration), coach_rows) if narration else None,
    }
    (corpus / "learned.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["corpus"]))
    print("rank key             ch-stressed% stressed% mentioned% downplayed% most-important-hits")
    for r in coach_rows:
        print(f"{r['rank']:>2} {r['key']:<15} {100 * r['channel_stressed_share']:9.1f} "
              f"{100 * r['stressed_share']:9.1f} {100 * r['mention_share']:9.1f} "
              f"{100 * r['downplayed_share']:9.1f} {r['most_important_hits']:>4}")
    for topic, t in by_topic.items():
        top = ", ".join(f"{r['key']} {100 * r['channel_stressed_share']:.0f}%" for r in t["rows"][:6])
        print(f"topic {topic} ({t['videos']} videos, {t['channels']} channels): {top}")
    print("old narration:", result["old_narration"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
