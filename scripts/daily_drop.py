#!/usr/bin/env python3
"""
daily_drop.py — the 7-minute daily dose: 8 flashcards + 1 full interview drill.

Selection is a PURE FUNCTION of (Deck/deck.json, Deck/schedule.json, date). That matters:
the GitHub Actions cron that pushes the Telegram message runs this in read-only mode and
never writes schedule.json, so there is exactly one writer (you, via --grade or the
platform's grade export). Days still rotate without any state write: unseen cards sit in a
fixed global order and the drop takes a window that advances 8 places per day, so an
ungraded week still never replays a card.

Usage:
  python3 scripts/daily_drop.py                    # today's drop to stdout
  python3 scripts/daily_drop.py --date 2026-08-20  # a specific day
  python3 scripts/daily_drop.py --dry-run 7        # preview the next 7 days
  python3 scripts/daily_drop.py --md               # also write Daily/YYYY-MM-DD.md
  python3 scripts/daily_drop.py --html             # also write Deck/daily.html (mirror)
  python3 scripts/daily_drop.py --telegram         # push (needs TELEGRAM_BOT_TOKEN/CHAT_ID)
  python3 scripts/daily_drop.py --grade "1g 2a 3e 4g 5g 6f 7x 8e d:partial"
  python3 scripts/daily_drop.py --bury c-bw-grpo:"not my lever"   # retire a card
  python3 scripts/daily_drop.py --buried                          # what is retired, and why

Grades: a=again (reset to box 1) · g=good (box+1) · e=easy (box+2) · f=again, alias.
        x=reject — retires the card permanently (same as --bury).
Drill grades: d:nailed | d:partial | d:blank
"""

import argparse
import html
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DECK_JSON = BASE / "Deck" / "deck.json"
SCHEDULE = BASE / "Deck" / "schedule.json"
DAILY_DIR = BASE / "Daily"
HTML_OUT = BASE / "Deck" / "daily.html"

CARDS_PER_DROP = 8
REVIEW_CAP = 4          # at most this many due-review cards, rest are new material
THEME_CAP = 3           # no more than this many cards from one theme in a drop
BOX_DAYS = {0: 0, 1: 1, 2: 3, 3: 7, 4: 21, 5: 60}
MAX_BOX = 5
TIER_CYCLE = [1, 2, 3, 4]
# One reserved slot each per 8-card block. aryaka-ops is here because a card about the
# SASE support copilot is the one most likely to change what gets built this week;
# left to the round-robin it would surface once every ten days.
FLAVOUR_THEMES = ("fails-vs-works", "buzzword-decoder", "framing-questions")
EPOCH = date(2026, 8, 18)   # day 1 of the habit
MIRROR = os.environ.get("DECK_MIRROR_URL",
                        "https://rishabh-parihar-1612.github.io/ai-briefings-platform/")
TG_LIMIT = 3800          # Telegram hard cap is 4096; leave headroom for entities


# ----------------------------------------------------------------- state


def load_deck():
    if not DECK_JSON.exists():
        sys.exit("Deck/deck.json missing — run: python3 scripts/build_deck.py")
    return json.loads(DECK_JSON.read_text())


def load_schedule():
    if SCHEDULE.exists():
        return json.loads(SCHEDULE.read_text())
    return {"version": 1, "cards": {}, "drills": {}, "history": {}}


def save_schedule(state):
    SCHEDULE.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def seed(*parts) -> int:
    """FNV-1a 32-bit. Deliberately not sha256: the platform's JS Today view has to
    reproduce this exact ordering, and FNV-1a ports to JS in four lines."""
    h = 0x811C9DC5
    for b in "::".join(str(p) for p in parts).encode():
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def days_between(a: str, b: date) -> int:
    return (b - datetime.strptime(a, "%Y-%m-%d").date()).days


# ------------------------------------------------------------- selection


def interleave(items, key_of, first=FLAVOUR_THEMES, block=CARDS_PER_DROP):
    """Lay the unseen cards out in blocks of `block`, each block reserving one slot per
    `first` theme and round-robining the rest across the remaining themes.

    Two properties fall out, and the daily drop depends on both. Any window of `block`
    consecutive cards contains one card from each flavour theme — so a contrast card and a
    buzzword card are in every drop structurally, not by a post-hoc swap. And the window
    the drop takes advances exactly `block` per day, so it stays aligned to these blocks
    and no card is served twice before the whole deck has cycled. The flavour guarantee in
    pick_cards is a safety net that should never fire; when it did fire it rotated on its
    own cycle, which is what used to serve one card twice in a single pass."""
    buckets: dict = {}
    for it in items:
        buckets.setdefault(key_of(it), []).append(it)
    lead = [k for k in first if buckets.get(k)]
    rest = sorted((k for k in buckets if k not in lead), key=lambda k: seed("deck", k))
    order, spin = [], 0
    while any(buckets.values()):
        for k in lead:                        # reserved slots, one per flavour theme
            if buckets.get(k):
                order.append(buckets[k].pop(0))
        for _ in range(max(0, block - len(lead))):
            live = [k for k in rest if buckets.get(k)] or [k for k in lead if buckets.get(k)]
            if not live:
                break
            k = live[spin % len(live)]
            spin += 1
            order.append(buckets[k].pop(0))
    return order


def pick_cards(deck, state, day: date):
    """Deterministic: same (deck, state, day) always yields the same 8 cards.

    Policy knob, deliberately in one place: REVIEW_CAP / THEME_CAP / CARDS_PER_DROP above.
    Raise REVIEW_CAP if you want the deck to consolidate instead of expand.
    """
    ds = day.isoformat()
    buried = set(state.get("buried", {}))
    cards = {c["id"]: c for c in deck["cards"] if c["id"] not in buried}
    cstate = state.get("cards", {})

    def overdue(cid):
        e = cstate[cid]
        return days_between(e["last"], day) - BOX_DAYS.get(e.get("box", 1), 1)

    due = sorted((cid for cid in cards if cid in cstate and cstate[cid].get("last")
                  and overdue(cid) >= 0),
                 key=lambda cid: (-overdue(cid), seed(ds, cid)))
    # Unseen cards walk a fixed global order in a window that advances 8 per day, so an
    # ungraded day still cannot replay a card until the whole unseen pool has cycled.
    fresh = interleave(sorted((cid for cid in cards
                               if cid not in cstate or not cstate[cid].get("last")),
                              key=lambda cid: seed("deck", cid)),
                       lambda cid: cards[cid].get("theme"))
    if fresh:
        off = ((day - EPOCH).days * CARDS_PER_DROP) % len(fresh)
        new = fresh[off:] + fresh[:off]
    else:
        new = []
    resting = sorted((cid for cid in cards if cid in cstate and cstate[cid].get("last")
                      and overdue(cid) < 0),
                     key=lambda cid: (overdue(cid), seed(ds, cid)))

    chosen, themes, used = [], Counter(), set()

    def take(pool, limit):
        for cid in pool:
            if len(chosen) >= limit:
                return
            if cid in used:
                continue
            theme = cards[cid].get("theme")
            if themes[theme] >= THEME_CAP:
                continue
            chosen.append(cid)
            used.add(cid)
            themes[theme] += 1

    take(due, min(REVIEW_CAP, CARDS_PER_DROP))
    take(new, CARDS_PER_DROP)
    take(due, CARDS_PER_DROP)        # no new material left? keep reviewing
    take(resting, CARDS_PER_DROP)    # early days: pull cards forward rather than short-drop

    # flavour guarantee: every drop carries one card from each reserved theme — a
    # fails-vs-works contrast, a buzzword decode, and an applied Aryaka card.
    # Keyed on theme rather than kind: aryaka-ops has no kind of its own, and for the
    # other two the validator already pins kind to theme one-to-one.
    for theme in FLAVOUR_THEMES:
        if any(cards[cid].get("theme") == theme for cid in chosen):
            continue
        # same rotating window as the main pool: a date-seeded pick would happily
        # repeat yesterday's card, which is exactly what the guarantee must not do
        pool = sorted((cid for cid in cards if cards[cid].get("theme") == theme
                       and cid not in used),
                      key=lambda cid: (cid in cstate and bool(cstate[cid].get("last")),
                                       seed("deck", cid)))
        if not pool:
            continue
        pick = pool[(day - EPOCH).days % len(pool)]
        if len(chosen) >= CARDS_PER_DROP:
            # Evict from the back, but never a card already satisfying a flavour slot.
            # With two reserved themes a plain chosen[-1] was harmless; with three, the
            # third substitution silently overwrote the second one's pick.
            victim = next((i for i in range(len(chosen) - 1, -1, -1)
                           if cards[chosen[i]].get("theme") not in FLAVOUR_THEMES), None)
            if victim is None:
                continue
            used.discard(chosen[victim])
            chosen[victim] = pick
        else:
            chosen.append(pick)
        used.add(pick)

    return [cards[cid] for cid in chosen[:CARDS_PER_DROP]]


def pick_drill(deck, state, day: date):
    ds = day.isoformat()
    tier = TIER_CYCLE[(day - EPOCH).days % len(TIER_CYCLE)]
    dstate = state.get("drills", {})
    buried = set(state.get("buried", {}))
    live = [d for d in deck["drills"] if d["id"] not in buried]
    pool = [d for d in live if d.get("tier") == tier] or live
    if not pool:
        return None
    return sorted(pool, key=lambda d: (dstate.get(d["id"], {}).get("last", ""),
                                       seed(ds, d["id"])))[0]


def drop_for(deck, state, day: date):
    ds = day.isoformat()
    hist = state.get("history", {}).get(ds)
    if hist:
        by_id = {c["id"]: c for c in deck["cards"]}
        dr_by_id = {d["id"]: d for d in deck["drills"]}
        cards = [by_id[c] for c in hist.get("cards", []) if c in by_id]
        return {"date": ds, "day": (day - EPOCH).days + 1, "cards": cards,
                "drill": dr_by_id.get(hist.get("drill")), "replayed": True}
    return {"date": ds, "day": (day - EPOCH).days + 1,
            "cards": pick_cards(deck, state, day),
            "drill": pick_drill(deck, state, day), "replayed": False}


# -------------------------------------------------------------- renderers


def cite_of(rec):
    c = rec.get("cite")
    return " · ".join(c) if isinstance(c, list) else str(c or "")


def gap_note(rec, brackets=True):
    if not rec.get("gap"):
        return ""
    return " [modeled — ungrounded]" if brackets else " (modeled — ungrounded)"


def render_text(drop) -> str:
    out = [f"🎴 Daily Drop · {drop['date']} · day {drop['day']}", ""]
    for i, c in enumerate(drop["cards"], 1):
        out.append(f"{i}. {c['front']}")
        out.append(f"   → {c['back']}")
        if c.get("hook"):
            out.append(f"   ⚡ {c['hook']}")
        if c.get("aryaka"):
            out.append(f"   🏢 At Aryaka: {c['aryaka']}")
        out.append(f"   ({cite_of(c)}{gap_note(c)})")
    d = drop["drill"]
    if d:
        out += ["", f"🥊 Drill · tier {d['tier']} · {d.get('level','')} — {d['title']}", "",
                d["problem"], "", "Answer skeleton:"]
        out += [f"  {i}. {b}" for i, b in enumerate(d.get("skeleton") or [], 1)]
        if d.get("numbers"):
            out += ["", "Numbers:"] + [f"  · {n}" for n in d["numbers"]]
        if d.get("traps"):
            out += ["", "Traps:"] + [f"  · {t}" for t in d["traps"]]
        if d.get("followups"):
            out += ["", "Follow-ups:"]
            for fu in d["followups"]:
                out += [f"  Q: {fu['q']}", f"  A: {fu['a']}"]
        if d.get("termdrops"):
            out += ["", "Term drops: " + ", ".join(d["termdrops"])]
        out += ["", f"({cite_of(d)})"]
    out += ["", f"grade it:  python3 scripts/daily_drop.py --grade \"1g 2g 3a … d:partial\""]
    return "\n".join(out)


def render_md(drop) -> str:
    out = [f"# Daily Drop — {drop['date']} (day {drop['day']})", "",
           "8 cards + 1 drill, ~7 minutes. Answer before expanding.", "",
           "## Cards", ""]
    for i, c in enumerate(drop["cards"], 1):
        out += [f"**{i}. {c['front']}**", "", "<details><summary>Show answer</summary>", "",
                c["back"]]
        if c.get("hook"):
            out += ["", f"*Hook:* {c['hook']}"]
        if c.get("aryaka"):
            out += ["", f"*At Aryaka:* {c['aryaka']}"]
        wiki = f" · *Wiki:* [[{c['topic']}]]" if c.get("topic") else ""
        out += ["", f"*Source: {cite_of(c)}.*{gap_note(c, False)}{wiki}", "</details>", ""]
    d = drop["drill"]
    if d:
        out += [f"## Drill — {d['title']}", "",
                f"`{d['id']}` · tier {d['tier']} · {d.get('level','')}", "",
                d["problem"], ""]
        if d.get("assesses"):
            out += ["**Being graded on:** " + "; ".join(d["assesses"]), ""]
        out += ["<details><summary>Show answer</summary>", "", "**Skeleton**", ""]
        out += [f"{i}. {b}" for i, b in enumerate(d.get("skeleton") or [], 1)]
        for label, key in (("Numbers", "numbers"), ("Traps", "traps")):
            if d.get(key):
                out += ["", f"**{label}**", ""] + [f"- {x}" for x in d[key]]
        if d.get("followups"):
            out += ["", "**Follow-ups**", ""]
            for fu in d["followups"]:
                out += [f"- *{fu['q']}* — {fu['a']}"]
        if d.get("termdrops"):
            out += ["", "**Term drops:** " + ", ".join(f"`{t}`" for t in d["termdrops"])]
        links = " · ".join(f"[[{t}]]" for t in (d.get("topics") or []))
        out += ["", f"*Source: {cite_of(d)}.*" + (f" · *Wiki:* {links}" if links else ""),
                "</details>", ""]
    return "\n".join(out)


def render_telegram(drop):
    """Return a list of HTML messages, each under the Telegram size cap.

    ONE CARD PER MESSAGE, deliberately. Telegram clients reveal every <tg-spoiler> in a
    message from a single tap, so batching the eight cards into one message meant the first
    reveal dumped all eight answers and the recall test was over before it started. The
    size cap was never the binding constraint here; the spoiler's blast radius is.
    """
    e = html.escape
    n = len(drop["cards"])
    msgs = []

    for i, c in enumerate(drop["cards"], 1):
        hook = f"\n⚡ {e(c['hook'])}" if c.get("hook") else ""
        # the applied line goes INSIDE the spoiler: it is part of the answer, and seeing
        # "at Aryaka, the tunnel-flap runbook…" before recall gives the card away
        ary = f"\n🏢 <b>At Aryaka:</b> {e(c['aryaka'])}" if c.get("aryaka") else ""
        head = f"🎴 <b>Daily Drop · {drop['date']}</b> · day {drop['day']}\n\n" if i == 1 else ""
        msgs.append(f"{head}<b>{i}/{n}. {e(c['front'])}</b>\n"
                    f"<tg-spoiler>{e(c['back'])}{hook}{ary}</tg-spoiler>\n"
                    f"<i>{e(cite_of(c))}{e(gap_note(c))}</i>")

    d = drop["drill"]
    if d:
        head = (f"🥊 <b>Drill · tier {d['tier']} · {e(str(d.get('level','')))}</b>\n"
                f"<b>{e(d['title'])}</b>\n\n{e(d['problem'])}\n")
        body = ["<b>Skeleton</b>", "<tg-spoiler>"]
        body += [f"{i}. {e(b)}" for i, b in enumerate(d.get("skeleton") or [], 1)]
        if d.get("numbers"):
            body += ["", "Numbers: " + e(" | ".join(d["numbers"]))]
        if d.get("traps"):
            body += ["", "Traps: " + e(" | ".join(d["traps"]))]
        if d.get("followups"):
            body += [""] + [f"Q: {e(fu['q'])}\nA: {e(fu['a'])}" for fu in d["followups"]]
        if d.get("termdrops"):
            body += ["", "Term drops: " + e(", ".join(d["termdrops"]))]
        body += ["</tg-spoiler>", f"<i>{e(cite_of(d))}</i>",
                 f'<a href="{MIRROR}#/drill/{d["id"]}">open in Drill Room</a>']
        block = head + "\n".join(body)
        if len(block) > TG_LIMIT:
            trimmed = head + "\n".join(["<b>Skeleton</b>", "<tg-spoiler>"] +
                                       [f"{i}. {e(b)}" for i, b in
                                        enumerate(d.get("skeleton") or [], 1)] +
                                       ["</tg-spoiler>"])
            block = trimmed[:TG_LIMIT - 120] + \
                f'\n<a href="{MIRROR}#/drill/{d["id"]}">full drill →</a>'
        msgs.append(block)
    return msgs


def render_html(drop) -> str:
    e = html.escape
    cards = "".join(
        f'<div class="c"><div class="f">{i}. {e(c["front"])}</div>'
        f'<div class="b" hidden>{e(c["back"])}'
        + (f'<div class="h">⚡ {e(c["hook"])}</div>' if c.get("hook") else "")
        + (f'<div class="ary">🏢 <b>At Aryaka:</b> {e(c["aryaka"])}</div>'
           if c.get("aryaka") else "")
        + f'<div class="s">{e(cite_of(c))}{e(gap_note(c))}</div></div>'
        f'<button>reveal</button></div>'
        for i, c in enumerate(drop["cards"], 1))
    d = drop["drill"] or {}
    drill = ""
    if d:
        sk = "".join(f"<li>{e(b)}</li>" for b in d.get("skeleton") or [])
        extra = ""
        for label, key in (("Numbers", "numbers"), ("Traps", "traps")):
            if d.get(key):
                extra += f"<p><b>{label}</b></p><ul>" + \
                    "".join(f"<li>{e(x)}</li>" for x in d[key]) + "</ul>"
        if d.get("followups"):
            extra += "<p><b>Follow-ups</b></p><ul>" + "".join(
                f"<li><i>{e(fu['q'])}</i><br>{e(fu['a'])}</li>" for fu in d["followups"]) + "</ul>"
        drill = (f'<div class="c drill"><div class="f">🥊 tier {d.get("tier")} · '
                 f'{e(str(d.get("level","")))} — {e(d.get("title",""))}</div>'
                 f'<pre>{e(d.get("problem",""))}</pre>'
                 f'<div class="b" hidden><ol>{sk}</ol>{extra}'
                 f'<div class="s">{e(cite_of(d))}</div></div><button>reveal</button></div>')
    return f"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Daily Drop {drop['date']}</title>
<style>
:root{{color-scheme:dark}}
body{{background:#0e1116;color:#e6e6e6;font:16px/1.5 -apple-system,system-ui,sans-serif;
margin:0;padding:18px;max-width:720px}}
h1{{font-size:18px;margin:0 0 14px}}
.c{{background:#161b22;border:1px solid #262d38;border-radius:10px;padding:12px;margin:10px 0}}
.f{{font-weight:600}} .b{{margin-top:8px;color:#cfd6e0}}
.h{{margin-top:6px;color:#7ee787}} .s{{margin-top:8px;color:#8b949e;font-size:12.5px}}
.ary{{margin-top:8px;padding:7px 9px;border-left:3px solid #d29922;background:#1c1810;border-radius:0 4px 4px 0;color:#e3b341;font-size:13.5px}}
button{{margin-top:10px;background:#21262d;color:#58a6ff;border:1px solid #30363d;
border-radius:7px;padding:6px 12px;font:inherit;font-size:13px}}
pre{{white-space:pre-wrap;font:inherit;color:#cfd6e0;margin:8px 0}}
.drill{{border-color:#3d3320}} ol,ul{{padding-left:20px;margin:6px 0}}
</style>
<h1>🎴 Daily Drop · {drop['date']} · day {drop['day']}</h1>
{cards}{drill}
<script>
document.querySelectorAll('.c button').forEach(b=>b.addEventListener('click',()=>{{
  const body=b.previousElementSibling; body.hidden=!body.hidden;
  b.textContent=body.hidden?'reveal':'hide';}}));
</script>"""


# --------------------------------------------------------------- telegram


def send_telegram(msgs):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in the environment.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i, text in enumerate(msgs, 1):
        # one card per message means ~10 messages a day, so only the first is allowed to
        # make a sound; the rest arrive silently in the same thread
        payload = urllib.parse.urlencode({
            "chat_id": chat, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": "true",
            "disable_notification": "false" if i == 1 else "true"}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=payload)) as r:
                ok = json.loads(r.read()).get("ok")
            print(f"  telegram {i}/{len(msgs)}: {'sent' if ok else 'rejected'} ({len(text)} chars)")
        except urllib.error.HTTPError as err:
            body = err.read().decode()[:300]
            sys.exit(f"telegram message {i} failed: HTTP {err.code} — {body}")


# ----------------------------------------------------------------- grading


GRADE_MAP = {"a": "again", "f": "again", "g": "good", "e": "easy"}
# 'x' is not a grade — it retires the card. A card you cannot kill is a card you learn to
# skim past, which quietly poisons every box interval it appears in.
BURY_LETTER = "x"


def bury(state, card_id: str, ds: str, reason: str = ""):
    """Retire a card from every future drop.

    Kept in schedule.json rather than deleted from Deck/cards/ on purpose: the card stays
    in the corpus with its citation intact, and `build_deck.py` reports the buried set —
    so a rejected card reads as a bug report about the card, not as content quietly
    vanishing. Un-bury with --unbury.
    """
    state.setdefault("buried", {})[card_id] = {"date": ds, "reason": reason}
    state.get("cards", {}).pop(card_id, None)      # drop its box state; it is out of rotation


def apply_grades(deck, state, day: date, spec: str):
    drop = drop_for(deck, state, day)
    ds = day.isoformat()
    cards, drill = drop["cards"], drop["drill"]
    logged = []
    dgrade = None
    for token in spec.replace(",", " ").split():
        if token.startswith("d:"):
            dgrade = token[2:]
            continue
        idx, _, letter = token[:-1], None, token[-1].lower()
        if not idx.isdigit() or letter not in (set(GRADE_MAP) | {BURY_LETTER}):
            sys.exit(f"cannot parse grade token '{token}' — want e.g. 3g, 5a, 2x, d:partial")
        i = int(idx) - 1
        if not 0 <= i < len(cards):
            sys.exit(f"grade token '{token}' is outside 1-{len(cards)}")
        card = cards[i]
        if letter == BURY_LETTER:
            bury(state, card["id"], ds, "rejected during grading")
            logged.append((card["id"], "buried", None))
            continue
        entry = state["cards"].setdefault(card["id"], {"box": 0, "seen": 0, "again": 0})
        grade = GRADE_MAP[letter]
        if grade == "again":
            entry["box"] = 1
            entry["again"] = entry.get("again", 0) + 1
        else:
            entry["box"] = min(MAX_BOX, max(1, entry.get("box", 0)) + (2 if grade == "easy" else 1))
        entry["last"] = ds
        entry["seen"] = entry.get("seen", 0) + 1
        entry["grade"] = grade
        logged.append((card["id"], grade, entry["box"]))

    if dgrade and drill:
        e = state["drills"].setdefault(drill["id"], {"seen": 0})
        e["last"], e["grade"] = ds, dgrade
        e["seen"] = e.get("seen", 0) + 1

    state["history"][ds] = {"cards": [c["id"] for c in cards],
                            "drill": drill["id"] if drill else None,
                            "graded": True}
    save_schedule(state)
    return logged, (drill, dgrade)


def stats(deck, state):
    boxes = Counter(e.get("box", 0) for e in state.get("cards", {}).values())
    total = len(deck["cards"])
    touched = len(state.get("cards", {}))
    return (f"cards {touched}/{total} seen · boxes " +
            " ".join(f"b{b}:{boxes[b]}" for b in sorted(boxes)) +
            f" · drills {len(state.get('drills', {}))}/{len(deck['drills'])} attempted" +
            f" · {len(state.get('history', {}))} days logged")


def main():
    p = argparse.ArgumentParser(description="8 cards + 1 drill, every day, free.")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--dry-run", type=int, metavar="N", help="preview N days, write nothing")
    p.add_argument("--md", action="store_true", help="write Daily/<date>.md")
    p.add_argument("--html", action="store_true", help="write Deck/daily.html")
    p.add_argument("--telegram", action="store_true", help="push to Telegram")
    p.add_argument("--grade", metavar='"1g 2a …"',
                   help="grade today's drop and advance boxes; 'x' retires a card (e.g. 2x)")
    p.add_argument("--bury", metavar="ID[:reason]", action="append", default=[],
                   help="retire a card or drill by id so it never appears again")
    p.add_argument("--unbury", metavar="ID", action="append", default=[],
                   help="bring a retired card back into rotation")
    p.add_argument("--buried", action="store_true", help="list what is retired, and why")
    p.add_argument("--stats", action="store_true", help="print scheduler state only")
    args = p.parse_args()

    deck = load_deck()
    state = load_schedule()
    day = datetime.strptime(args.date, "%Y-%m-%d").date()

    if args.stats:
        print(stats(deck, state))
        return

    if args.bury or args.unbury or args.buried:
        ds = day.isoformat()
        ids = {c["id"] for c in deck["cards"]} | {d["id"] for d in deck["drills"]}
        for spec in args.bury:
            cid, _, reason = spec.partition(":")
            if cid not in ids:
                sys.exit(f"no card or drill with id '{cid}'")
            bury(state, cid, ds, reason or "rejected")
            print(f"  buried {cid}" + (f" — {reason}" if reason else ""))
        for cid in args.unbury:
            if state.get("buried", {}).pop(cid, None):
                print(f"  unburied {cid} — back in rotation as new material")
            else:
                print(f"  {cid} was not buried")
        if args.bury or args.unbury:
            save_schedule(state)
        b = state.get("buried", {})
        if args.buried or b:
            print(f"\n{len(b)} retired:")
            for cid, meta in sorted(b.items()):
                print(f"  {cid:<44} {meta.get('date','')}  {meta.get('reason','')}")
            if b:
                print("\nA retired card is a bug report — fix or delete it in Deck/cards/, "
                      "and build_deck.py lists these so they do not just vanish quietly.")
        return

    if args.grade:
        logged, (drill, dgrade) = apply_grades(deck, state, day, args.grade)
        for cid, grade, box in logged:
            if grade == "buried":
                print(f"  {cid}: retired — will not appear again (--unbury to undo)")
                continue
            nxt = BOX_DAYS[box]
            print(f"  {cid}: {grade} → box {box} (next in {nxt}d)")
        if dgrade and drill:
            print(f"  {drill['id']}: {dgrade}")
        print("\n" + stats(deck, state))
        print("\nDrill grades also belong in Quizzes/attempts.json — run /daily --grade "
              "through Claude if you want the attempts log and Wiki/_index.md refreshed.")
        return

    if args.dry_run:
        for i in range(args.dry_run):
            d = day + timedelta(days=i)
            drop = drop_for(deck, state, d)
            themes = Counter(c.get("theme") for c in drop["cards"])
            print(f"{drop['date']} day {drop['day']:>3}  drill {drop['drill']['id'] if drop['drill'] else '—':<34}"
                  f" cards: " + ", ".join(c["id"].replace("c-", "") for c in drop["cards"]))
            print(f"{'':>16}  themes: " + " ".join(f"{k}:{v}" for k, v in themes.most_common()))
        return

    drop = drop_for(deck, state, day)
    print(render_text(drop))
    if args.md:
        DAILY_DIR.mkdir(exist_ok=True)
        out = DAILY_DIR / f"{drop['date']}.md"
        out.write_text(render_md(drop))
        print(f"\nwrote {out.relative_to(BASE)}")
    if args.html:
        HTML_OUT.write_text(render_html(drop))
        print(f"wrote {HTML_OUT.relative_to(BASE)}")
    if args.telegram:
        send_telegram(render_telegram(drop))


if __name__ == "__main__":
    main()
