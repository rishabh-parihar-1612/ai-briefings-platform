#!/usr/bin/env python3
"""radar_terms.py — term momentum and lineage, which is the part worth reading.

A list of nine links is noise. What is actually useful is: *which vocabulary is gaining
ground, what practice does it replace, and what decision does it change.* That question is
answerable deterministically up to the prose:

  detection   which candidate terms appear, how often, across how many distinct sources
  momentum    mentions in a window + source diversity, vs. the term's first-seen date
  novelty     is this already in the glossary / already a deck card, or genuinely new
  lineage     which known succession chain does it extend (seeded from the corpus)

Only the prose needs a model. So the digest becomes a worksheet with the lineage slots
already named, and it is emitted ONLY when something crosses the bar — a quiet week
produces nothing rather than a daily list of links nobody reads.

State lives in Radar/terms.json (the ledger) and Radar/lineage.json (the chains).
"""

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# --- the bar. Below this, nothing is emitted.
RISING_MENTIONS = 3         # mentions inside the window
RISING_SOURCES = 2          # from at least this many distinct sources
# The general n-gram layer catches terms that do not announce themselves with a familiar
# suffix, which is the point — but it is noisier, so it has to clear a higher bar.
NGRAM_MENTIONS = 4
NGRAM_SOURCES = 3
WINDOW_DAYS = 14

# Discipline-naming patterns. These are how a new practice announces itself — the
# "prompt engineering -> context engineering -> harness engineering" succession is the
# shape to catch, so the patterns look for the naming, not for the topic.
PATTERNS = [
    re.compile(r"\b([a-z][a-z0-9]{2,18})[ -]engineering\b", re.I),
    re.compile(r"\b([a-z][a-z0-9]{2,18})[ -]driven[ -](?:development|design|dev)\b", re.I),
    re.compile(r"\b([a-z][a-z0-9]{2,18})[ -]ops\b", re.I),
    re.compile(r"\b(agentic[ -][a-z]{3,15})\b", re.I),
    re.compile(r"\b([a-z]{3,15}[ -](?:harness|loop|budget|gateway|cascade|orchestration))\b", re.I),
    re.compile(r"\b([A-Za-z]{2,12}RAG)\b"),
    re.compile(r"\b(context[ -][a-z]{3,15})\b", re.I),
    re.compile(r"\b(test[ -]time[ -][a-z]{3,15}|inference[ -]time[ -][a-z]{3,15})\b", re.I),
    re.compile(r"\b([a-z]{3,15}[ -]as[ -]a[ -][a-z]{3,10})\b", re.I),
    re.compile(r"\b(world[ -]models?|spec[ -]kit|agent[ -]cards?)\b", re.I),
]
ACRONYM = re.compile(r"\b([A-Z][0-9A-Z]{1,5})\b")   # allows A2A, S3, GPT4

# Product-shaped names: AgentCore, FrontierCode, LazyGraphRAG, LangGraph. An internal
# capital is the giveaway, and these are how new tools arrive before they have a category.
CAMEL = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-zA-Z]+)+)\b")

# The naming patterns above are high-precision but narrow — a term does not have to be
# called "X engineering" to matter. So general bigrams and trigrams are extracted too, and
# kept honest by two gates: the ITEM must be about AI at all (AI_CONTEXT), and the phrase
# must not be built from filler (NGRAM_STOP). Momentum then filters whatever survives.
AI_CONTEXT = re.compile(
    r"\b(agent|agentic|llm|model|inference|prompt|token|context window|rag|retrieval|"
    r"embedding|eval|evaluation|fine[- ]tun\w+|reasoning|autonomous|multimodal|"
    r"transformer|attention|hallucinat\w+|guardrail|orchestrat\w+|copilot|"
    r"mcp|tool[- ]call\w*|memory|vector|benchmark|training|distill\w+|quantiz\w+)\b", re.I)
NGRAM_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "this", "that", "these",
    "those", "with", "without", "for", "from", "into", "onto", "over", "under", "about",
    "how", "why", "what", "when", "where", "who", "which", "your", "our", "their", "its",
    "is", "are", "was", "were", "be", "been", "being", "can", "will", "would", "should",
    "could", "may", "might", "must", "do", "does", "did", "have", "has", "had", "get",
    "gets", "got", "make", "makes", "made", "use", "uses", "used", "using", "new", "now",
    "more", "most", "less", "very", "just", "also", "one", "two", "three", "first", "last",
    "next", "best", "better", "good", "great", "big", "small", "here", "there", "all",
    "any", "some", "no", "not", "you", "we", "they", "it", "he", "she", "i", "my", "me",
    "at", "by", "in", "on", "of", "to", "as", "up", "out", "so", "such", "via", "per",
    "announcing", "introducing", "launch", "launches", "launched", "post", "blog", "read",
    "learn", "guide", "tutorial", "part", "week", "day", "today", "year", "years",
}
NGRAM_MIN_WORD = 3            # every word in a kept n-gram must be at least this long

# The item being about AI is not enough — "between turns" passed that gate. The phrase
# itself has to name something in the domain. Broad on purpose: this is not the
# "X engineering" pattern, it is "does this phrase belong to this field at all".
ANCHOR = {
    "agent", "agents", "agentic", "llm", "llms", "model", "models", "inference", "prompt",
    "prompts", "token", "tokens", "context", "window", "rag", "retrieval", "retriever",
    "embedding", "embeddings", "vector", "eval", "evals", "evaluation", "judge", "judges",
    "reasoning", "reason", "autonomy", "autonomous", "memory", "compaction", "cache",
    "caching", "harness", "loop", "loops", "guardrail", "guardrails", "orchestration",
    "orchestrator", "trajectory", "budget", "budgets", "sandbox", "protocol", "router",
    "routing", "tool", "tools", "benchmark", "distillation", "quantization", "compute",
    "multimodal", "transformer", "attention", "hallucination", "grounding", "groundedness",
    "checkpoint", "checkpointing", "state", "policy", "planner", "planning", "supervisor",
    "workflow", "workflows", "pipeline", "corpus", "provenance", "injection", "tenancy",
    "copilot", "mcp", "a2a", "fine-tuning", "finetuning", "training", "serving",
    "throughput", "latency", "batching", "decoding", "sampling", "reward", "alignment",
}

# Acronyms that are noise in this domain, or too generic to track.
ACRONYM_STOP = {
    "AI", "ML", "API", "APIS", "LLM", "LLMS", "GPU", "CPU", "HTTP", "HTTPS", "JSON", "YAML",
    "SQL", "CLI", "SDK", "IDE", "URL", "PDF", "CSV", "AWS", "GCP", "IBM", "USA", "EU", "UK",
    "CEO", "CTO", "CIO", "COO", "VP", "HR", "PR", "QA", "OK", "TL", "DR", "FAQ", "RSS",
    "GA", "SOTA", "RC", "LTS", "OSS", "MIT", "BSD", "GPL", "NEW", "AND", "THE", "FOR",
    "YOU", "ALL", "NOT", "USE", "GET", "SET", "RUN", "TOP", "NOW", "HOW", "WHY",
}
# Vendors and products are not practices. Tracking "NVIDIA" as a rising term is exactly
# the noise this ledger exists to avoid — a company shipping things is not a trend.
VENDOR_STOP = {
    "NVIDIA", "OPENAI", "ANTHROPIC", "GOOGLE", "DEEPMIND", "MICROSOFT", "AZURE", "META",
    "MISTRAL", "COHERE", "GITHUB", "GITLAB", "CLOUDFLARE", "DATADOG", "SNOWFLAKE",
    "DATABRICKS", "SALESFORCE", "SAP", "ORACLE", "APPLE", "AMAZON", "SAGEMAKER",
    "BEDROCK", "VERTEX", "COPILOT", "CHATGPT", "CLAUDE", "GEMINI", "LLAMA", "QWEN",
    "DEEPSEEK", "GROK", "GPT", "GPT4", "GPT5", "SLM", "SLMS", "VLM", "MOE", "PYTORCH", "TENSORFLOW", "HUGGINGFACE", "LANGCHAIN", "LLAMAINDEX",
    "TEMPORAL", "REDIS", "POSTGRES", "KAFKA", "KUBERNETES", "DOCKER", "TERRAFORM",
}
PHRASE_STOP = {
    "prompt engineering",       # tracked as an established chain head, not as news
    "software engineering", "machine engineering", "data engineering", "platform engineering",
    "reverse engineering", "social engineering", "value engineering",
}

# Successions the corpus already documents. The head of each chain is what "where is this
# going" means concretely: when a new term extends a chain, the digest can say it is the
# Nth name for the same practice and name what actually changed each time.
DEFAULT_LINEAGE = {
    "chains": [
        {
            "id": "instruction-to-runtime",
            "question": "Where does the engineering effort live?",
            "steps": [
                {"term": "prompt engineering", "era": "2022-2023",
                 "what_changed": "How you phrased a single request",
                 "cite": "doc: 2026-08-19-buzzword-terms §1"},
                {"term": "context engineering", "era": "2024-2025",
                 "what_changed": "Which information enters a finite window per step",
                 "cite": "issue 004, §14"},
                {"term": "harness engineering", "era": "2026",
                 "what_changed": "The deterministic runtime around the model: memory schema, "
                                 "loop policy, tool contracts, budgets, trace",
                 "cite": "doc: agent-harness-loop-llmops-deep-dive §1"},
                {"term": "loop engineering", "era": "2026",
                 "what_changed": "Iteration and termination policy specifically — which of the "
                                 "seven termination mechanisms ended the run",
                 "cite": "doc: agent-harness-loop-llmops-deep-dive §24"},
            ],
        },
        {
            "id": "eval-maturity",
            "question": "What does 'it works' mean?",
            "steps": [
                {"term": "vibes-driven development", "era": "2023",
                 "what_changed": "It seemed better than last week",
                 "cite": "doc: agentic-ai-system-design-deep-dive §17"},
                {"term": "eval-driven development", "era": "2024-2025",
                 "what_changed": "A regression eval set gates the deploy",
                 "cite": "issue 006, §7"},
                {"term": "trajectory evaluation", "era": "2025-2026",
                 "what_changed": "The path is graded, not only the answer",
                 "cite": "doc: agentic-ai-evaluation-definitive-reference §5"},
            ],
        },
        {
            "id": "authoring-shift",
            "question": "What artifact do humans actually review?",
            "steps": [
                {"term": "vibe coding", "era": "2025",
                 "what_changed": "Review the generated diff, or don't",
                 "cite": "doc: agentic-ai-system-design-deep-dive §4"},
                {"term": "spec-driven development", "era": "2026",
                 "what_changed": "Review a versioned spec; the code is regenerable output",
                 "cite": "doc: 2026-08-19-buzzword-terms §5"},
            ],
        },
    ]
}


def norm(t: str) -> str:
    return re.sub(r"[\s-]+", " ", t.strip().lower())


def ngrams(text: str, n: int):
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text.lower())
    for i in range(len(words) - n + 1):
        chunk = words[i:i + n]
        if any(w in NGRAM_STOP for w in chunk):
            continue
        if any(len(w) < NGRAM_MIN_WORD for w in chunk):
            continue
        if all(w.upper() in VENDOR_STOP for w in chunk):
            continue
        if not any(w in ANCHOR or w.rstrip("s") in ANCHOR for w in chunk):
            continue
        yield " ".join(chunk)


def extract_kinds(text: str, general: bool = True):
    """Same as extract(), but returns {term: layer} so rising() can hold the noisier
    general-n-gram layer to a higher bar than the high-precision patterns."""
    kinds = {}
    for pat in PATTERNS:
        for m in pat.finditer(text):
            phrase = norm(m.group(0))
            if phrase in PHRASE_STOP or len(phrase) < 5:
                continue
            kinds[phrase] = "pattern"
    for m in ACRONYM.finditer(text):
        a = m.group(1)
        if a.upper() not in ACRONYM_STOP and a.upper() not in VENDOR_STOP:
            kinds.setdefault(a, "acronym")
    for m in CAMEL.finditer(text):
        name = m.group(1)
        if name.upper() not in VENDOR_STOP and len(name) > 5:
            kinds.setdefault(name, "product")
    if general:
        for t in extract(text) - set(kinds):
            kinds[t] = "ngram"
    return kinds


def extract(text: str, general: bool = True):
    """Candidate terms from one item. Deterministic, no model.

    Three layers, in decreasing precision: the discipline-naming patterns, product-shaped
    CamelCase names and acronyms, then general bigrams/trigrams gated on the item actually
    being about AI. The last layer is what catches a term that does not announce itself
    with a familiar suffix — most new vocabulary does not.
    """
    found = set()
    for pat in PATTERNS:
        for m in pat.finditer(text):
            phrase = norm(m.group(0))
            if phrase in PHRASE_STOP or len(phrase) < 5:
                continue
            found.add(phrase)
    for m in ACRONYM.finditer(text):
        a = m.group(1)
        if a.upper() not in ACRONYM_STOP and a.upper() not in VENDOR_STOP:
            found.add(a)          # acronyms keep their case; they are the signal
    for m in CAMEL.finditer(text):
        name = m.group(1)
        if name.upper() not in VENDOR_STOP and len(name) > 5:
            found.add(name)
    in_domain = bool(AI_CONTEXT.search(text)) or any(
        w in ANCHOR for w in re.findall(r"[a-z][a-z0-9-]*", text.lower()))
    if general and in_domain:
        for n in (2, 3):
            for g in ngrams(text, n):
                if g not in PHRASE_STOP and len(g) > 8:
                    found.add(g)
    return found


def load_ledger(path: Path):
    return json.loads(path.read_text()) if path.exists() else {"terms": {}}


def load_lineage(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_LINEAGE, indent=1))
    return DEFAULT_LINEAGE


def known_terms(vocab, deck_path: Path):
    """Vocabulary the corpus already covers — used to separate 'gaining ground' from
    'new to me', which is the distinction that decides whether a card is needed."""
    known = {norm(t) for t in vocab.get("terms", [])}
    known |= {norm(s.replace("-", " ")) for s in vocab.get("topics", [])}
    # a topic slug's leading phrase counts as known too: "mcp" from "mcp", "semantic cache"
    # from "semantic-cache", so a two-word term is not reported as novel
    known |= {norm(" ".join(s.split("-")[:2])) for s in vocab.get("topics", [])}
    if deck_path.exists():
        for c in json.loads(deck_path.read_text()).get("cards", []):
            known.add(norm(c.get("front", "")))
            for w in re.findall(r"\b[a-z][a-z -]{4,30}\b", (c.get("front", "") + " " +
                                                            c.get("back", "")).lower()):
                if w.strip() and len(w.strip()) > 6:
                    known.add(norm(w))
    return known


def update(ledger, items, today: str):
    """Fold today's items into the term ledger."""
    for it in items:
        text = f"{it.get('title','')} {it.get('summary','')}"
        for term, kind in extract_kinds(text).items():
            e = ledger["terms"].setdefault(term, {"first_seen": today, "mentions": [],
                                                  "kind": kind})
            e.setdefault("kind", kind)
            if any(m["url"] == it.get("url") for m in e["mentions"]):
                continue
            e["mentions"].append({"date": it.get("date", today), "source": it.get("source", "?"),
                                  "title": it.get("title", "")[:160], "url": it.get("url", "")})
            e["last_seen"] = today
    return ledger


def prune(ledger, today: str, keep_days: int = 45):
    """Drop one-off terms that never repeated. Without this the ledger grows without bound
    once general n-grams are extracted, and every stale singleton is a term that can never
    reach the bar but still costs storage and read time."""
    cutoff = (datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone.utc)
              - timedelta(days=keep_days)).date().isoformat()
    dropped = 0
    for term in list(ledger["terms"]):
        e = ledger["terms"][term]
        if len(e.get("mentions", [])) <= 1 and (e.get("last_seen") or e.get("first_seen", "")) < cutoff:
            del ledger["terms"][term]
            dropped += 1
    return dropped


def momentum(entry, today: str, window: int = WINDOW_DAYS):
    cutoff = (datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone.utc)
              - timedelta(days=window)).date().isoformat()
    recent = [m for m in entry.get("mentions", []) if m.get("date", "") >= cutoff]
    return len(recent), len({m["source"].split(" (")[0] for m in recent}), recent


def rising(ledger, today: str, known, decoded):
    """Terms that crossed the bar and are not already taught by the deck."""
    out = []
    for term, e in ledger["terms"].items():
        n, srcs, recent = momentum(e, today)
        kind = e.get("kind", "ngram")
        need_n = NGRAM_MENTIONS if kind == "ngram" else RISING_MENTIONS
        need_s = NGRAM_SOURCES if kind == "ngram" else RISING_SOURCES
        if n < need_n or srcs < need_s:
            continue
        # compare normalized: the ledger keeps "MCP" for display, but the deck knows it as
        # "mcp" — comparing raw case reported already-taught terms as brand new
        key = norm(term)
        if key in decoded:
            continue
        out.append({"term": term, "mentions": n, "sources": srcs, "kind": kind,
                    "first_seen": e.get("first_seen"), "recent": recent,
                    "already_known": key in known})
    return sorted(out, key=lambda x: (-x["mentions"], -x["sources"]))


def chain_for(term, lineage):
    """Which documented succession does this term extend, if any."""
    t = norm(term)
    for ch in lineage.get("chains", []):
        for step in ch["steps"]:
            if norm(step["term"]) == t:
                return ch, step
        # a new "<x> engineering" extends the instruction-to-runtime chain by construction
        if ch["id"] == "instruction-to-runtime" and t.endswith(" engineering"):
            return ch, None
        if ch["id"] == "authoring-shift" and "driven development" in t:
            return ch, None
    return None, None


def decoded_terms(deck_path: Path):
    """Terms the buzzword decoder already covers — these are momentum, not news."""
    if not deck_path.exists():
        return set()
    out = set()
    for c in json.loads(deck_path.read_text()).get("cards", []):
        if c.get("kind") == "buzzword":
            out.add(norm(c.get("front", "").split("—")[0].split("?")[0]))
            out.add(norm(c["id"].replace("c-bw-", "").replace("-", " ")))
    return out
