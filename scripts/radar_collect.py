#!/usr/bin/env python3
"""radar_collect.py — the free, deterministic half of the radar loop.

Runs unattended in GitHub Actions (no LLM, no API key, no quota). It fetches the
sources, drops what it has already seen, and — the part that makes it useful without a
model — scores every surviving item against YOUR vocabulary: the wiki topic slugs, the
glossary terms, and the topics your deck already covers. So the digest says "this touches
semantic-cache, which you have 3 cards on" rather than just handing you a headline.

Deliberately zero-dependency (stdlib only) and deliberately fail-soft: a dead feed is
REPORTED, never silently skipped, because a collector that goes quiet looks identical to
a week with no news.

  python3 scripts/radar_collect.py --out Radar/inbox            # collect + write digest
  python3 scripts/radar_collect.py --days 3 --dry-run           # see what it would write
  python3 scripts/radar_collect.py --vocab Deck/mirror/vocab.json

The decode step (prose, buzzword four-beats, inflection mapping) is NOT here — it needs a
model. `--decode <provider>` is a documented hook that stays a no-op until one is
available; see decode_stub() at the bottom.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_terms as rt                                     # noqa: E402

BASE = Path(__file__).resolve().parent.parent
UA = "ai-briefings-radar/1.0 (+personal learning system)"
TIMEOUT = 25

# Feeds: (label, url). All of these were probed live on 2026-08-19 and parse.
#
# Deliberately absent, because they have no working feed as of that date: Anthropic
# (anthropic.com/rss.xml, /rss, /news/rss.xml all 404), LangChain (301s to a JS page),
# Meta AI, Mistral, and DeepLearning.AI's The Batch. Vendor RSS is dying. Those vendors
# are covered instead by the HN queries below plus Simon Willison, who tracks them
# closely — losing a source silently is the failure mode this list exists to prevent, so
# when you add one back, add it here rather than working around it.
FEEDS = [
    ("OpenAI",           "https://openai.com/news/rss.xml"),
    ("Google DeepMind",  "https://deepmind.google/blog/rss.xml"),
    ("Cloudflare",       "https://blog.cloudflare.com/rss/"),
    ("GitHub",           "https://github.blog/feed/"),
    ("Simon Willison",   "https://simonwillison.net/atom/everything/"),
    ("Hugging Face",     "https://huggingface.co/blog/feed.xml"),
    ("AWS ML",           "https://aws.amazon.com/blogs/machine-learning/feed/"),
    ("Azure",            "https://azure.microsoft.com/en-us/blog/feed/"),
]

# Hacker News: one query per theme. These are tuned for the two things the deck wants —
# production incidents (which become contrast cards) and agent architecture practice.
HN_QUERIES = [
    "AI agent postmortem", "LLM outage", "agent production incident",
    "prompt injection", "agent harness", "context engineering",
    "RAG production", "LLM cost", "agent evaluation", "MCP server",
    # these three stand in for vendors with no working RSS feed
    "Anthropic Claude", "LangGraph", "agent framework",
]
HN_MIN_POINTS = 40          # below this, it is noise, not signal

ARXIV_QUERY = ("cat:cs.AI+AND+(abs:%22agent+harness%22+OR+abs:%22agentic%22+OR+"
               "abs:%22tool+use%22+OR+abs:%22LLM+agent%22)")

# Words that mark an item as a candidate for a fails-vs-works contrast card.
INCIDENT_WORDS = re.compile(
    r"\b(postmortem|post-mortem|incident|outage|regression|breach|leak(ed)?|"
    r"exfiltrat\w+|root cause|rollback|degrad\w+|failure|broke|deleted|"
    r"runaway|blew (up|through)|lesson\w* learned)\b", re.I)

# Words that mark an item as moving one of the strategic watches.
INFLECTION_WORDS = re.compile(
    r"\b(pricing|price cut|deprecat\w+|retir\w+|general availability|GA\b|launch\w*|"
    r"acqui\w+|funding|benchmark|state of the art|SOTA|open[- ]weight|"
    r"standard\w*|protocol|interop\w*|regulation|compliance|EU AI Act)\b", re.I)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/rss+xml, application/atom+xml, application/xml, application/json, */*"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def strip_html(s: str) -> str:
    txt = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()
    # arXiv titles carry LaTeX; "D$^2$ACCI" should read as "D2ACCI"
    return re.sub(r"\$+\^?\{?([^${}]*)\}?\$+", r"\1", txt)


def parse_date(s: str):
    if not s:
        return None
    s = s.strip()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            d = datetime.strptime(s.replace("GMT", "+0000"), fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_feed(label: str, raw: bytes):
    """RSS 2.0 and Atom, without a dependency. Returns [] rather than raising."""
    out = []
    root = ET.fromstring(raw)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        def g(*names):
            for n in names:
                el = it.find(n) if "}" not in n else it.find(n, ns)
                if el is None:
                    el = it.find(f"a:{n}", ns)
                if el is not None:
                    return (el.text or el.get("href") or "").strip()
            return ""
        link = g("link")
        if not link:
            el = it.find("a:link", ns)
            link = el.get("href", "") if el is not None else ""
        out.append({
            "source": label,
            "title": strip_html(g("title")),
            "url": link,
            "summary": strip_html(g("description", "summary", "content"))[:600],
            "date": (parse_date(g("pubDate", "published", "updated")) or
                     datetime.now(timezone.utc)).date().isoformat(),
        })
    return out


def collect_hn(days: int):
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    items, errs = [], []
    for q in HN_QUERIES:
        url = ("https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=20"
               f"&numericFilters=created_at_i>{since},points>{HN_MIN_POINTS}"
               f"&query={urllib.parse.quote(q)}")
        try:
            data = json.loads(fetch(url))
        except Exception as e:
            errs.append(f"HN '{q}': {type(e).__name__} {e}")
            continue
        for h in data.get("hits", []):
            link = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
            items.append({"source": f"HN ({h.get('points')} pts)", "title": h.get("title") or "",
                          "url": link, "summary": strip_html(h.get("story_text") or "")[:400],
                          "date": (h.get("created_at") or "")[:10],
                          "discussion": f"https://news.ycombinator.com/item?id={h.get('objectID')}"})
    return items, errs


def collect_arxiv(days: int, attempts: int = 3):
    """arXiv rate-limits hard and asks callers to space requests out; a 429 on the first
    try is normal, so back off rather than reporting a dead source."""
    url = ("https://export.arxiv.org/api/query?search_query=" + ARXIV_QUERY +
           "&sortBy=submittedDate&sortOrder=descending&max_results=25")
    last = ""
    for n in range(attempts):
        try:
            return parse_feed("arXiv", fetch(url)), []
        except Exception as e:
            last = f"{type(e).__name__} {e}"
            time.sleep(3 * (n + 1))
    return [], [f"arXiv: {last} (after {attempts} attempts)"]


def load_vocab(path: Path):
    """Topic slugs + glossary terms + which topics the deck already covers."""
    if path and path.exists():
        return json.loads(path.read_text())
    vocab = {"topics": [], "terms": [], "deck_topics": {}}
    td = BASE / "Wiki" / "topics"
    if td.exists():
        vocab["topics"] = sorted(f.stem for f in td.glob("*.md"))
    gl = BASE / "Wiki" / "glossary.md"
    if gl.exists():
        vocab["terms"] = sorted({m.group(1).lower() for m in
                                 re.finditer(r"^\*\*(.+?)\*\*", gl.read_text(), re.M)})
    dj = BASE / "Deck" / "deck.json"
    if dj.exists():
        counts = {}
        for c in json.loads(dj.read_text()).get("cards", []):
            if c.get("topic"):
                counts[c["topic"]] = counts.get(c["topic"], 0) + 1
        vocab["deck_topics"] = counts
    return vocab


# Short, high-signal tokens the len>4 filter would otherwise drop. Matched case-sensitively
# against the original text so "GA" does not fire on "organisation".
ACRONYMS = {
    "MCP": ["mcp"], "A2A": ["a2a"], "RAG": ["kg-rag", "embeddings-and-vector-databases"],
    "HITL": ["hitl-design"], "SLO": ["ai-slis-slos"], "SLI": ["ai-slis-slos"],
    "PII": ["multi-tenant-ai-security"], "KV": ["prompt-caching"],
    "TTL": ["semantic-cache"], "GRPO": [], "SDD": [],
}


def score(item, vocab):
    """Deterministic relevance: which of MY topics does this touch, and is it an
    incident or an inflection? No model involved — this is string matching against a
    vocabulary the corpus already defines."""
    raw = f"{item['title']} {item.get('summary','')}"
    text = raw.lower()
    hits = []
    for slug in vocab.get("topics", []):
        words = [w for w in slug.split("-") if len(w) > 3]
        if not words:
            continue
        # requiring EVERY word missed obvious matches (an MCP-and-guardrails post scored
        # zero against prompt-injection-defense). Two significant words, or the leading
        # phrase verbatim, is the balance that stopped false negatives without inviting noise.
        phrase = " ".join(slug.split("-")[:2])
        if phrase in text or sum(1 for w in words if w in text) >= min(2, len(words)):
            hits.append(slug)
    for acr, slugs in ACRONYMS.items():
        if re.search(rf"\b{acr}\b", raw):
            hits += [s for s in slugs if s in vocab.get("topics", [])]
            item.setdefault("acronyms", []).append(acr)
    terms = [t for t in vocab.get("terms", []) if len(t) > 4 and t in text]
    item["topics"] = sorted(set(hits))
    item["terms"] = sorted(set(terms))[:8]
    # Incident classification is for write-ups, not papers: an arXiv abstract saying
    # "root cause" or "failure" is describing a research problem, not a production
    # post-mortem, and letting those through is what fills the section with noise.
    is_paper = item["source"].startswith("arXiv")
    item["incident"] = bool(INCIDENT_WORDS.search(text)) and not is_paper
    item["paper"] = is_paper
    item["inflection"] = bool(INFLECTION_WORDS.search(text))
    item["covered"] = {t: vocab.get("deck_topics", {}).get(t, 0) for t in item["topics"]}
    item["score"] = (3 * len(item["topics"]) + len(item["terms"])
                     + 2 * len(item.get("acronyms", []))
                     + (4 if item["incident"] else 0) + (2 if item["inflection"] else 0))
    return item


def render(day, trend, incidents, movers, papers, errors, vocab):
    """The digest, rewritten around terms rather than articles.

    The old version was a list of links, which is noise. This one leads with the
    vocabulary that gained ground, places it on a documented succession chain, and names
    the five slots a decode has to fill. It is a worksheet, not a newsletter — and it is
    only written when something crossed the bar.
    """
    L = ["---", f"date: {day}", "collector: radar_collect.py", "decoded: false",
         f"rising_terms: {len(trend)}", f"incidents: {len(incidents)}",
         "status: pending-approval", "---", "",
         f"# Radar — {day}", "",
         "Terms that gained ground, not a list of articles. Detection is deterministic "
         "(mention counts, distinct sources, first-seen dates, and which succession chain "
         "the term extends); the prose is not written yet. Run `/radar --deck` to fill the "
         "slots and mine cards.", ""]
    if errors:
        L += ["> Sources that failed this run (reported, never hidden — a silent collector "
              "looks the same as a quiet week):", ""] + [f"> - {e}" for e in errors] + [""]

    if trend:
        L += ["## Terms gaining ground", ""]
        for t in trend:
            L += [f"### {t['term']}", "",
                  f"**Momentum:** {t['mentions']} mentions across {t['sources']} distinct "
                  f"sources in the last 14 days · first seen {t['first_seen']}",
                  f"**Already in my vocabulary:** {'yes — this is momentum, not news' if t['already_known'] else 'NO — new to the corpus'}"]
            ch, step = t.get("chain"), t.get("step")
            if ch:
                L += ["", f"**Extends a documented succession:** *{ch['question']}*", ""]
                for st in ch["steps"]:
                    mark = " ← **this one**" if step and st["term"] == step["term"] else ""
                    L += [f"- `{st['era']}` **{st['term']}** — {st['what_changed']} "
                          f"({st['cite']}){mark}"]
                if not step:
                    L += [f"- `2026` **{t['term']}** — ??? *(the decode must fill this in, "
                          f"and say whether it is a real shift or a rename)*"]
            else:
                L += ["", "**No known chain** — either it starts one, or it is a product name."]
            L += ["", "**Evidence:**"]
            for m in t["recent"][:3]:
                L += [f"- {m['source']} · {m['date']} · [{m['title']}]({m['url']})"]
            L += ["", "**Slots the decode must fill:**",
                  "1. What was the previous approach, concretely, and what did it fail at?",
                  "2. What changed mechanically — not in framing, in mechanism?",
                  "3. Which decision in our stack does it change, if any? (\"none\" is a valid answer "
                  "and a useful one.)",
                  "4. Real shift, or rebranding? Say which, with the reasoning.",
                  "5. Does it invalidate an existing card? Name the id. This matters more than "
                  "adding one — nothing else in the system retires stale cards.", ""]

    if incidents:
        L += ["## Incidents — candidate contrast cards", "",
              "*Matched incident vocabulary. The ❌/✅ card writes itself if the post names "
              "what they did and what fixed it.*", ""]
        for i in incidents:
            cov = ", ".join(f"{k} ({v} cards)" for k, v in i["covered"].items()) or "no topic match"
            L += [f"- **{i['title']}** — {i['source']} · {i['date']}",
                  f"  {i['url']}", f"  touches: {cov}"]
        L += [""]

    if movers:
        L += ["## Moves a strategic watch", "",
              "*Pricing, deprecation, GA, standards or regulation language against a topic "
              "we already track.*", ""]
        for i in movers:
            cov = ", ".join(i["topics"]) or "—"
            L += [f"- **{i['title']}** — {i['source']} · {i['date']}",
                  f"  {i['url']}", f"  touches: {cov}"]
        L += [""]

    if papers:
        L += ["## Research touching two or more of my topics", "",
              "*arXiv, filtered to papers that hit at least two tracked topics. Skim the "
              "abstract; these are for direction, not for cards.*", ""]
        for i in papers:
            L += [f"- **{i['title']}** — {i['date']} · {i['url']}",
                  f"  touches: {', '.join(i['topics'])}"]
        L += [""]

    L += ["## Bar for this digest existing at all", "",
          f"- a term with ≥{3} mentions from ≥{2} distinct sources in 14 days, not already "
          "decoded in the deck; or",
          "- an item matching incident vocabulary; or",
          "- a pricing/deprecation/GA/standards item touching a topic we track; or",
          "- a paper touching two or more tracked topics.", "",
          "Nothing else is written. A quiet week produces no file, no commit and no "
          "notification — that is deliberate, because a daily list of links trains you to "
          "ignore the channel.", ""]
    return "\n".join(L)


def decode_stub(items):
    """Deliberately unimplemented. GitHub Models was retired 2026-07-30 (the inference API
    now returns HTTP 410), Copilot coding agent consumes premium requests, and a Claude
    OAuth token in CI draws down the interactive subscription budget. When one of those
    becomes available, implement here: one request per item (any free tier will cap input
    around 8K tokens, so per-item is the right granularity), fail soft to decoded:false so
    a quota error never costs you the day's collection."""
    return items, ["decode skipped: no model available in CI (see decode_stub docstring)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="Radar/inbox", help="directory for the digest")
    ap.add_argument("--days", type=int, default=2, help="lookback window")
    ap.add_argument("--seen", default=None, help="path to seen.json (default: <out>/seen.json)")
    ap.add_argument("--vocab", default=None, help="pre-built vocab.json (for the mirror repo)")
    ap.add_argument("--min-score", type=int, default=1, help="drop items below this score")
    ap.add_argument("--ledger", default=None, help="term ledger json (default: <out>/terms.json)")
    ap.add_argument("--lineage", default=None, help="succession chains json (default: Radar/lineage.json)")
    ap.add_argument("--decode", default=None, help="model provider for the decode pass (unavailable)")
    ap.add_argument("--emit-vocab", metavar="PATH",
                    help="write vocab.json from the local corpus and exit (the mirror repo "
                         "has no Wiki/, so publish_platform.sh ships this file for it)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.emit_vocab:
        v = load_vocab(None)
        Path(args.emit_vocab).write_text(json.dumps(v, indent=1))
        print(f"vocab: {len(v['topics'])} topics, {len(v['terms'])} terms, "
              f"{len(v['deck_topics'])} deck-covered topics -> {args.emit_vocab}")
        return

    out_dir = BASE / args.out
    seen_path = Path(args.seen) if args.seen else out_dir / "seen.json"
    seen = set(json.loads(seen_path.read_text())) if seen_path.exists() else set()
    vocab = load_vocab(Path(args.vocab) if args.vocab else None)

    items, errors = [], []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).date().isoformat()
    for label, url in FEEDS:
        try:
            got = parse_feed(label, fetch(url))
            fresh = [i for i in got if i["date"] >= cutoff]
            items += fresh
            if not got:
                errors.append(f"{label}: parsed 0 items (feed shape changed?)")
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__} {e}")
    hn, e1 = collect_hn(args.days); items += hn; errors += e1
    ax, e2 = collect_arxiv(args.days); items += ax; errors += e2

    # dedupe by url, then against everything already collected on earlier days
    uniq, urls = [], set()
    for i in items:
        u = (i.get("url") or "").split("?")[0].rstrip("/")
        if not u or u in urls or u in seen:
            continue
        urls.add(u); i["url"] = u; uniq.append(i)

    scored = [score(i, vocab) for i in uniq]
    day = datetime.now(timezone.utc).date().isoformat()

    # --- term momentum: the ledger is the memory that makes "gaining ground" meaningful
    ledger_path = Path(args.ledger) if args.ledger else out_dir / "terms.json"
    lineage_path = Path(args.lineage) if args.lineage else BASE / "Radar" / "lineage.json"
    ledger = rt.load_ledger(ledger_path)
    lineage = rt.load_lineage(lineage_path)
    ledger = rt.update(ledger, scored, day)
    known = rt.known_terms(vocab, BASE / "Deck" / "deck.json")
    decoded = rt.decoded_terms(BASE / "Deck" / "deck.json")
    trend = rt.rising(ledger, day, known, decoded)
    for t in trend:
        ch, step = rt.chain_for(t["term"], lineage)
        t["chain"], t["step"] = ch, step

    incidents = [i for i in scored if i["incident"]]
    movers = [i for i in scored if i["inflection"] and not i["incident"]
              and i["topics"] and not i.get("paper")]
    papers = [i for i in scored if i.get("paper") and len(i["topics"]) >= 2]

    if args.decode:
        scored, derr = decode_stub(scored); errors += derr

    # --- the bar. Nothing crossed it means nothing is written.
    if not (trend or incidents or movers or papers):
        print(f"nothing crossed the bar ({len(scored)} new items seen, "
              f"{len(ledger['terms'])} terms tracked"
              + (f", {len(errors)} source errors" if errors else "") + ")")
        if not args.dry_run:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(ledger, indent=1))
            seen_path.write_text(json.dumps(sorted(seen | urls), indent=0))
        return

    body = render(day, trend, incidents, movers, papers, errors, vocab)
    if args.dry_run:
        print(body)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{day}.md").write_text(body)
    ledger_path.write_text(json.dumps(ledger, indent=1))
    seen_path.write_text(json.dumps(sorted(seen | urls), indent=0))
    bits = []
    if trend:
        bits.append(f"{len(trend)} term(s) gaining ground: " + ", ".join(t["term"] for t in trend[:3]))
    if incidents:
        bits.append(f"{len(incidents)} incident(s)")
    if papers:
        bits.append(f"{len(papers)} paper(s)")
    if movers:
        bits.append(f"{len(movers)} watch mover(s)")
    print("; ".join(bits) + (f"; {len(errors)} source errors" if errors else ""))


if __name__ == "__main__":
    main()
