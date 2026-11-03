#!/usr/bin/env python3
"""
Pull resolved binary Polymarket markets for the pre- vs post-boundary leakage study.
Drop-in replacement for scripts/legacy/pull_metaculus.py (same outputs, same column names), written
after Metaculus closed its API data to default accounts (see HANDOFF.md).

Groups (the boundary is OUR grouping date, not a claim about the model):
  post: resolved (closedTime) after --post-start AND scheduled deadline (endDate) more than 3 days
        after it. This reduces mis-grouping from late settlement; it does not guarantee the outcome
        was unknown at the boundary (a "has X happened?" market can resolve long after X happened).
  pre : resolved inside a window ending --pre-margin-days before --post-start.

Market baseline ("community prediction" column cp_prob): the YES-share price
--lead-days before resolution (daily price history from the CLOB API, last point at or
before the target time, never later than resolution). Resolution from outcomePrices
(["1","0"] = YES, ["0","1"] = NO; anything else, e.g. voided 50/50, is skipped).

Filters: binary Yes/No market, volume >= --min-volume USD, horizon (created -> resolved)
>= --min-horizon-days, not sports / crypto-price / parlay-style (tag + keyword filter),
and the market was already trading at the lead point.
Sampling: candidates are collected from several listing pages and --n-per-group are drawn
at random with --seed, instead of taking the most recent ones.

Outputs (in --out): questions.csv (keep away from the model), prompts.jsonl (feed to the
model), meta.json, raw/<id>.json (market record + price history).

Usage:
  python scripts/pull_polymarket.py --post-start 2026-07-01 --cutoff-label "2026-06 training data cutoff (Anthropic docs)" --count-only
  python scripts/pull_polymarket.py --post-start 2026-07-01 --cutoff-label "..." --n-per-group 15
"""
import argparse, csv, json, random, re, sys, time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

GAMMA = "https://gamma-api.polymarket.com/markets"
CP_MIN = 0.03   # drop markets the market itself had already called (price outside [0.03, 0.97] at the lead point)
MAX_PER_TOPIC = 2   # at most this many markets per group sharing a capitalised title word (e.g. 'Brazilian', 'Fed')
TOPIC_STOP = {'will', 'the', 'before', 'after', 'by', 'in', 'us', 'of', 'and', 'for', 'to', 'a', 'an', 'be', 'on', 'at',
              'no', 'any', 'or', 'with', 'next', 'new', 'out', 'as', 'president', 'election', 'win', 'wins', 'most', 'first',
              'second', 'round', 'seats', 'party', 'x', 'vs', 'meeting', 'end', 'change', 'rates', 'interest'}

def topic_tokens(title):
    toks = re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", title or '')
    return {t.lower() for t in toks if t[0].isupper() and t.lower() not in TOPIC_STOP}

CLOB = "https://clob.polymarket.com/prices-history"
UA = {"User-Agent": "leakage-takehome/0.2 (research take-home)"}

EXCLUDE_TAGS = {"sports", "esports", "nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc", "mma",
                "golf", "f1", "cricket", "boxing", "crypto", "crypto prices", "bitcoin", "ethereum",
                "solana", "memecoins", "stocks", "fdv", "hourlies", "games", "chess", "esport"}
EXCLUDE_RE = re.compile(
    r"\bvs\.?\b|\bO/U\b|\bSpread:|\bMoneyline|\bFDV\b|\bUp or Down\b|price of|above \$|below \$|"
    r"hit \$|reach \$|dip to \$|\bETF\b|market cap|\bFloor price|\bLargest company\b|"
    r"\bNBA\b|\bNFL\b|\bNHL\b|\bMLB\b|\bWNBA\b|\bNCAA\b|\bUFC\b|\bF1\b|\bPGA\b|\bATP\b|\bWTA\b|\bFIFA\b|"
    r"Premier League|La Liga|Serie A|Bundesliga|Ligue 1|Champions League|Europa League|Super Bowl|World Cup|"
    r"\bFinals\b|Stanley Cup|World Series|Conference\b|Drivers' Champion|Constructors|Ballon d'Or|goalscorer|"
    r"Grand Slam|Wimbledon|US Open|Roland Garros|Australian Open|\bMasters\b|Olympic|\bmedal|\bHeisman|\bMVP\b|"
    r"playoffs?|relegat|\bwins? the\b.*\b(?:League|Cup|Finals|Bowl|Championship|Open|Classic|Derby|Series|Title)\b|"
    r"Crude Oil|\bWTI\b|\bBrent\b|\bGold\b|\bSilver\b|S&P|Nasdaq|\bDow\b|all-time high|\bhit \(|"
    r"play for|sign with|traded to|next team|\bLoL\b|\bWorlds\b|Valorant|\bCS2\b|\bDota\b|esports?|boxing|\bfight\b|\bbout\b|"
    r"\bChampion\b|\bcoach\b|\bdraft\b|\bgoals?\b|\bpoints\b|\bassists\b|\brebounds\b|\bweather\b|temperature|\bhighest temp",
    re.I)
# matched against the event title + slug only
COMBO_RE = re.compile(r"\bcombo\b|\bparlay\b|\bMSI\b|\bLEC\b|\bLCK\b|\bLPL\b|\blol-|\btransfer\b|"
                      r"Mid-Season Invitational|Biletnikoff|\bUEFA\b|Women's Euro|\b(?:NL|AL) (?:East|West|Central)\b|\bDivision\b|tornado|hurricane|all[ -]time high|StarLadder|\bBLAST\b|\bIEM\b|Budapest Major|\bMajor 20\d\d\b|\bNFC\b|\bAFC\b|Final Four|March Madness|precipitation|rainfall|snowfall|\bscoring\b|\bthis season\b|at least \$|\bIndex\b", re.I)


def parse_ts(s):
    if not s:
        return None
    s = str(s).replace("Z", "+00:00").replace(" ", "T")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        return None


def get(s, url, params, retries=5):
    for i in range(retries):
        try:
            r = s.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  network error {e}; retrying", file=sys.stderr); time.sleep(2 ** i); continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** (i + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {url} {params}")


def list_markets(s, params, max_pages, page_size=200):
    for page in range(max_pages):
        try:
            data = get(s, GAMMA, {**params, "limit": page_size, "offset": page * page_size})
        except requests.HTTPError as e:
            # Gamma rejects offsets beyond ~2000 with 422: treat as the end of this listing, but only
            # once we are past the first page (a 422 on page 0 is a bad parameter and must fail loudly)
            if e.response is not None and e.response.status_code == 422 and page > 0:
                print(f"  listing stops at offset {page * page_size} (422 offset cap); "
                      f"use --slice-days to see more", file=sys.stderr)
                return
            raise
        if not data:
            return
        yield from data
        if len(data) == page_size and page == max_pages - 1:
            print(f"  WARNING: listing {params.get('end_date_min')}..{params.get('end_date_max')} still full after "
                  f"{max_pages} pages; markets beyond offset {max_pages * page_size} were not seen "
                  f"(raise --max-pages or lower --slice-days)", file=sys.stderr)
        time.sleep(0.5)


def date_slices(start, end, slice_days):
    """Split [start, end] into consecutive end_date windows so each listing stays under the offset cap.
    Windows share their boundary instant (no gap); a market listed in two windows is dropped by the
    per-group `seen` set in collect()."""
    if not slice_days:
        return [(start, end)]
    out, t = [], start
    while t < end:
        u = min(t + timedelta(days=slice_days), end)
        out.append((t, u)); t = u
    return out


def tags_of(m):
    out = set()
    for t in (m.get("tags") or []):
        out.add(str(t.get("label") or t.get("slug") or "").lower())
    for ev in (m.get("events") or []):
        for t in (ev.get("tags") or []):
            out.add(str(t.get("label") or t.get("slug") or "").lower())
    return out


def basic_ok(m, min_volume):
    """Cheap checks on the listing record. Returns (ok, reason)."""
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
    except json.JSONDecodeError:
        return False, "bad json"
    if [o.lower() for o in outcomes] != ["yes", "no"]:
        return False, "not yes/no"
    if prices not in (["1", "0"], ["0", "1"]):
        return False, f"unresolved prices {prices}"
    if m.get("umaResolutionStatus") not in (None, "resolved"):
        return False, f"uma status {m.get('umaResolutionStatus')}"
    if float(m.get("volumeNum") or 0) < min_volume:
        return False, "low volume"
    if m.get("sportsMarketType") or m.get("gameId"):
        return False, "sports field"
    tags = tags_of(m)
    if tags & EXCLUDE_TAGS:
        return False, "excluded tag"
    # Check the event title/slug too: combo ("parlay") markets, esports series and transfer markets
    # often carry the tell-tale word only there (e.g. 'Fall Elections Combo', 'lol-msi-2026', 'transfer').
    ev_text = " ".join(f"{ev.get('title') or ''} {ev.get('slug') or ''}" for ev in (m.get("events") or []))
    q_text = m.get("question") or ""
    if EXCLUDE_RE.search(q_text) or EXCLUDE_RE.search(ev_text) or COMBO_RE.search(q_text) or COMBO_RE.search(ev_text):
        return False, "excluded keyword"
    return True, None


def times_of(m):
    open_t = parse_ts(m.get("createdAt") or m.get("startDate"))
    resolve_t = parse_ts(m.get("closedTime") or m.get("umaEndDate"))
    sched_end = parse_ts(m.get("endDate"))
    return open_t, resolve_t, sched_end


def price_at(s, token_id, target):
    data = get(s, CLOB, {"market": token_id, "interval": "max", "fidelity": 1440})
    hist = data.get("history") or []
    best = None
    for h in hist:
        t = datetime.fromtimestamp(h["t"], timezone.utc)
        if t <= target:
            best = (t, float(h["p"]))
    return best, hist


def extract(s, m, lead_days, min_horizon_days):
    open_t, resolve_t, sched_end = times_of(m)
    if not (open_t and resolve_t):
        return None, "missing times", None
    horizon = (resolve_t - open_t).total_seconds() / 86400
    if horizon < min_horizon_days:
        return None, "short horizon", None
    target = resolve_t - timedelta(days=lead_days)
    if target <= open_t:
        return None, "lead point before open", None
    try:
        token = json.loads(m.get("clobTokenIds") or "[]")[0]
    except (json.JSONDecodeError, IndexError):
        return None, "no token id", None
    best, hist = price_at(s, token, target)
    if best is None or (target - best[0]).days > 3:
        return None, "no price near lead point", hist
    if not (CP_MIN <= best[1] <= 1 - CP_MIN):
        return None, f"lead-point price outside [{CP_MIN}, {1-CP_MIN}]", hist
    prices = json.loads(m["outcomePrices"])
    res = 1 if prices == ["1", "0"] else 0
    # First daily price at or after creation (createdAt), at most 7 days later; blank otherwise. This is
    # the baseline closest to the model's "today = open_time", but it is NOT the price at the instant of
    # creation (daily points, thin early books), so cp_prob (lead point) stays the primary control.
    open_pt = None
    for h in sorted(hist, key=lambda h: h["t"]):
        t = datetime.fromtimestamp(h["t"], timezone.utc)
        if t >= open_t:
            if t - open_t <= timedelta(days=7):
                open_pt = (t, float(h["p"]))
            break
    row = {
        "id": int(m["id"]),
        "url": f"https://polymarket.com/market/{m.get('slug')}",
        "title": m.get("question"),
        "description": "",
        "resolution_criteria": (m.get("description") or "").strip(),
        "fine_print": "",
        "open_time": open_t.isoformat(),
        "close_time": (sched_end or resolve_t).isoformat(),   # scheduled end date (safe to show the model)
        "resolve_time": resolve_t.isoformat(),
        "horizon_days": round(horizon, 1),
        "cp_time": best[0].isoformat(), "cp_prob": round(best[1], 4),
        "cp_open_time": open_pt[0].isoformat() if open_pt else "",
        "cp_open_prob": round(open_pt[1], 4) if open_pt else "",
        "cp_forecasters": round(float(m.get("volumeNum") or 0)),   # USD volume stands in for forecaster count
        "resolution": res,
        "tags": "|".join(sorted(tags_of(m))),
    }
    return row, None, hist


def collect(s, group, list_params, keep, n_want, lead_days, min_horizon, min_volume, raw_dir,
            max_pages, rng, count_only, seen, keep_ids=(), max_per_topic=MAX_PER_TOPIC):
    """list_params: one params dict or a list of them (date slices), listed one after another.
    keep_ids: market ids from an earlier, smaller sample that are placed first (if they still pass
    every filter) so the new sample extends the old one instead of redrawing it."""
    cands, skipped, tagc = [], Counter(), Counter()
    slices = list_params if isinstance(list_params, list) else [list_params]
    for m in (m for lp in slices for m in list_markets(s, lp, max_pages)):
        if m.get("id") in seen:
            continue
        seen.add(m.get("id"))
        open_t, resolve_t, sched_end = times_of(m)
        if resolve_t is None or not keep(open_t, resolve_t, sched_end):
            continue
        ok, why = basic_ok(m, min_volume)
        if not ok:
            skipped[why] += 1
            continue
        if (resolve_t - open_t).total_seconds() / 86400 < min_horizon:
            skipped["short horizon"] += 1
            continue
        for t in tags_of(m):
            tagc[t] += 1
        cands.append(m)
    by_event = {}
    kept_ids = {str(i) for i in keep_ids}
    for m in cands:
        ev = ((m.get('events') or [{}])[0]).get('id') or f"m{m['id']}"
        cur = by_event.get(ev)
        # one market per event: the highest-volume one, except that a market from the earlier sample
        # keeps its slot so --keep really extends that sample
        if cur is None or (str(m.get('id')) in kept_ids and str(cur.get('id')) not in kept_ids) or \
           (str(cur.get('id')) not in kept_ids and float(m.get('volumeNum') or 0) > float(cur.get('volumeNum') or 0)):
            by_event[ev] = m
    n_before = len(cands); cands = list(by_event.values())
    print(f"  {group}: {n_before} candidates passed listing filters, {len(cands)} after one-per-event dedupe; "
          f"skipped={dict(skipped)}", file=sys.stderr)
    if count_only:
        for m in cands[:40]:
            print(f"     cand {m['id']} {(m.get('closedTime') or '')[:10]} vol={round(float(m.get('volumeNum') or 0))} {m.get('question')[:70]}", file=sys.stderr)
        return []
    rng.shuffle(cands)
    keep_ids = {str(i) for i in keep_ids}
    if keep_ids:
        kept = [m for m in cands if str(m.get('id')) in keep_ids]
        missing = keep_ids - {str(m.get('id')) for m in kept}
        print(f"  {group}: {len(kept)}/{len(keep_ids)} earlier-sample ids found among candidates"
              + (f"; not found (dropped): {sorted(missing)}" if missing else ""), file=sys.stderr)
        cands = kept + [m for m in cands if str(m.get('id')) not in keep_ids]
    rows, skipped2, topic_count = [], Counter(), Counter()
    for m in cands:
        toks = topic_tokens(m.get('question'))
        if any(topic_count[t] >= max_per_topic for t in toks):
            skipped2['topic cap'] += 1
            continue
        row, why, hist = extract(s, m, lead_days, min_horizon)
        time.sleep(0.3)
        if row is None:
            skipped2[why] += 1
            continue
        row["group"] = group
        for t in toks:
            topic_count[t] += 1
        rows.append(row)
        (raw_dir / f"{m['id']}.json").write_text(json.dumps({"market": m, "price_history": hist}, indent=1))
        print(f"  [{group}] {len(rows):2d}/{n_want} id={m['id']} cp={row['cp_prob']:.2f} res={row['resolution']} "
              f"h={row['horizon_days']}d vol={row['cp_forecasters']} {row['title'][:55]}", file=sys.stderr)
        if len(rows) >= n_want:
            break
    if skipped2:
        print(f"  {group}: detail-stage skips {dict(skipped2)}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-start", required=True, help="grouping boundary YYYY-MM-DD")
    ap.add_argument("--cutoff-label", default="", help="the model's OFFICIAL cutoff, stored verbatim in meta.json")
    ap.add_argument("--n-per-group", type=int, default=15)
    ap.add_argument("--lead-days", type=int, default=7)
    ap.add_argument("--pre-margin-days", type=int, default=120)
    ap.add_argument("--pre-window-days", type=int, default=365)
    ap.add_argument("--min-volume", type=float, default=20000, help="USD traded volume")
    ap.add_argument("--min-horizon-days", type=float, default=14)
    ap.add_argument("--max-pages", type=int, default=10, help="listing pages of 200 per group (and per slice)")
    ap.add_argument("--slice-days", type=int, default=0,
                    help="split each group's end_date range into windows of this many days; the Gamma listing "
                         "caps offset at ~2000, so slicing is how to see more than ~2000 markets per group")
    ap.add_argument("--max-per-topic", type=int, default=MAX_PER_TOPIC,
                    help="max markets per group sharing a capitalised title word")
    ap.add_argument("--keep", default="", help="questions.csv of an earlier sample; its ids are placed first "
                                              "so the new sample extends it (ids must still pass all filters)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    boundary = datetime.strptime(a.post_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pre_end = boundary - timedelta(days=a.pre_margin_days)
    pre_start = pre_end - timedelta(days=a.pre_window_days)
    out = Path(a.out); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers.update(UA)
    rng = random.Random(a.seed)
    seen = {"post": set(), "pre": set()}   # per group: the two groups' end_date ranges overlap, so a market
                                           # dropped by one group's keep() must still be visible to the other

    def keep_post(o, r, e=None):
        # resolved after the boundary AND the question's own deadline is after it, so the
        # outcome was not already determined (e.g. 'by June 30?' resolving on July 1)
        return r > boundary and (e is None or e > boundary + timedelta(days=3))

    def keep_pre(o, r, e=None):
        return pre_start <= r <= pre_end

    keep_ids = {"post": set(), "pre": set()}
    if a.keep:
        for r in csv.DictReader(open(a.keep)):
            keep_ids.setdefault(r["group"], set()).add(r["id"])
        print(f"KEEP: {len(keep_ids['post'])} post + {len(keep_ids['pre'])} pre ids from {a.keep}", file=sys.stderr)

    base = {"closed": "true", "order": "volumeNum", "ascending": "false", "volume_num_min": a.min_volume}
    fmt = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")   # full timestamp: the API treats a bare date as midnight
    # end_date (the market's scheduled end) ranges; keep_* then filters on the actual resolution time.
    now = datetime.now(timezone.utc)
    post_params = [{**base, "end_date_min": fmt(lo), "end_date_max": fmt(hi)}
                   for lo, hi in date_slices(boundary - timedelta(days=1), now + timedelta(days=400), a.slice_days)]
    if not a.slice_days:
        post_params = [{**base, "end_date_min": fmt(boundary - timedelta(days=1))}]
    pre_params = [{**base, "end_date_min": fmt(lo), "end_date_max": fmt(hi)}
                  for lo, hi in date_slices(pre_start, pre_end + timedelta(days=400), a.slice_days)]
    print(f"POST: resolved after {boundary.date()} ({len(post_params)} listing slice(s))", file=sys.stderr)
    post = collect(s, "post", post_params,
                   keep_post, a.n_per_group, a.lead_days, a.min_horizon_days, a.min_volume, raw,
                   a.max_pages, rng, a.count_only, seen["post"], keep_ids["post"], a.max_per_topic)
    print(f"PRE: resolved in [{pre_start.date()}, {pre_end.date()}] ({len(pre_params)} listing slice(s))", file=sys.stderr)
    pre = collect(s, "pre", pre_params,
                  keep_pre, a.n_per_group, a.lead_days, a.min_horizon_days, a.min_volume, raw,
                  a.max_pages, rng, a.count_only, seen["pre"], keep_ids["pre"], a.max_per_topic)

    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": "polymarket",
            "model_cutoff_label": a.cutoff_label, "post_start_boundary": boundary.date().isoformat(),
            "pre_window": [pre_start.date().isoformat(), pre_end.date().isoformat()],
            "lead_days": a.lead_days, "min_volume_usd": a.min_volume,
            "min_horizon_days": a.min_horizon_days, "seed": a.seed, "cp_min": CP_MIN, "max_per_topic": a.max_per_topic,
            "slice_days": a.slice_days, "max_pages": a.max_pages, "extends_sample": a.keep or None,
            "n_post": len(post), "n_pre": len(pre)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if a.count_only:
        print("count-only run; see candidate counts above", file=sys.stderr)
        return
    post_ids = {r["id"] for r in post}
    pre = [r for r in pre if r["id"] not in post_ids]
    rows = post + pre
    if not rows:
        sys.exit("no markets collected")
    with open(out / "questions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out / "prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "group", "title", "description",
                    "resolution_criteria", "fine_print", "open_time", "close_time")}) + "\n")
    for g, sub in (("post", post), ("pre", pre)):
        if sub:
            base_rate = sum(r["resolution"] for r in sub) / len(sub)
            cpb = sum((r["cp_prob"] - r["resolution"]) ** 2 for r in sub) / len(sub)
            hz = sum(r["horizon_days"] for r in sub) / len(sub)
            print(f"{g}: n={len(sub)} base_rate(yes)={base_rate:.2f} market_brier={cpb:.3f} "
                  f"mean_horizon={hz:.0f}d", file=sys.stderr)
    print(f"wrote {out/'questions.csv'}, {out/'prompts.jsonl'}, {out/'meta.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
