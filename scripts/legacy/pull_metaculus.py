#!/usr/bin/env python3
"""
Pull resolved binary Metaculus questions for a pre- vs post-boundary leakage study.

Groups (the boundary is OUR grouping date, not a claim about the model):
  post: resolved after --post-start AND still open for forecasts after --post-start
        (so the outcome was not yet known at the boundary; Metaculus closes questions
        once the outcome is known)
  pre : resolved inside a window ending --pre-margin-days before --post-start
        ("possible training exposure", not "guaranteed seen")

For each question we record the community prediction (CP) at a fixed point before
resolution (--lead-days before actual resolution, never later than close time), the
resolution, and the forecast horizon.

Outputs (in --out):
  questions.csv   everything incl. labels          (keep away from the model)
  prompts.jsonl   id + text only                   (feed this to the model)
  meta.json       all parameters + model cutoff label, for the write-up
  raw/<id>.json   raw API responses

Usage:
  python scripts/legacy/pull_metaculus.py --post-start 2026-07-01 --cutoff-label "2026-06 (training data cutoff, Anthropic docs)" --count-only
  python scripts/legacy/pull_metaculus.py --post-start 2026-07-01 --cutoff-label "..." --n-per-group 15
  METACULUS_TOKEN=xxxx python scripts/legacy/pull_metaculus.py ...   (optional, raises rate limits)
"""
import argparse, csv, json, os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE = "https://www.metaculus.com/api"
UA = {"User-Agent": "leakage-takehome/0.2 (research take-home)"}


def parse_ts(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def session():
    s = requests.Session()
    s.headers.update(UA)
    tok = os.environ.get("METACULUS_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Token {tok}"
    return s


def get(s, url, params=None, retries=4):
    for i in range(retries):
        r = s.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = 2 ** (i + 1)
            print(f"  429 rate-limited, sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {url}")


def list_posts(s, published_after, published_before, max_pages=60, page_size=50):
    params = {"statuses": "resolved", "forecast_type": "binary",
              "order_by": "-published_at", "limit": page_size, "offset": 0}
    if published_after:
        params["published_at__gt"] = published_after.strftime("%Y-%m-%d")
    if published_before:
        params["published_at__lt"] = published_before.strftime("%Y-%m-%d")
    for _ in range(max_pages):
        data = get(s, f"{BASE}/posts/", params)
        results = data.get("results", [])
        if not results:
            return
        yield from results
        if not data.get("next"):
            return
        params["offset"] += page_size
        time.sleep(1.0)


def times_of(post):
    q = post.get("question") or {}
    return (parse_ts(q.get("open_time") or post.get("published_at")),
            parse_ts(q.get("actual_close_time") or q.get("scheduled_close_time")),
            parse_ts(q.get("actual_resolve_time") or q.get("resolved_at")))


def cp_at(history, t):
    best = None
    for h in history:
        st, et = parse_ts(h.get("start_time")), parse_ts(h.get("end_time"))
        if st is None or st > t or (et is not None and et <= t):
            continue
        best = h
    if best is None:
        return None, None
    for key in ("centers", "means", "forecast_values"):
        v = best.get(key)
        if v:
            p = v[-1] if key == "forecast_values" else v[0]   # forecast_values=[P(no),P(yes)]
            return float(p), int(best.get("forecaster_count") or 0)
    return None, None


def extract(detail, lead_days, min_forecasters):
    q = detail.get("question") or detail
    if q.get("type") != "binary":
        return None, "not binary"
    res = (q.get("resolution") or "").lower()
    if res not in ("yes", "no"):
        return None, f"resolution={res!r}"
    open_t, close_t, resolve_t = times_of(detail)
    if not (resolve_t and close_t and open_t):
        return None, "missing times"
    target = min(resolve_t - timedelta(days=lead_days), close_t - timedelta(minutes=1))
    if target <= open_t:
        return None, "lead point before open"
    aggs = q.get("aggregations") or {}
    hist = (aggs.get("recency_weighted") or aggs.get("unweighted") or {}).get("history") or []
    if not hist:
        return None, "no CP history (with_cp=true missing?)"
    p, n = cp_at(hist, target)
    if p is None:
        return None, "no CP at target time"
    if n is not None and n < min_forecasters:
        return None, f"<{min_forecasters} forecasters"
    return {
        "id": detail.get("id"),
        "url": f"https://www.metaculus.com/questions/{detail.get('id')}/",
        "title": detail.get("title") or q.get("title"),
        "description": q.get("description") or "",
        "resolution_criteria": q.get("resolution_criteria") or "",
        "fine_print": q.get("fine_print") or "",
        "open_time": open_t.isoformat(), "close_time": close_t.isoformat(),
        "resolve_time": resolve_t.isoformat(),
        "horizon_days": round((resolve_t - open_t).total_seconds() / 86400, 1),
        "cp_time": target.isoformat(), "cp_prob": round(p, 4), "cp_forecasters": n,
        "resolution": 1 if res == "yes" else 0,
    }, None


def collect(s, group, pub_after, pub_before, keep, n_want, lead_days,
            min_forecasters, raw_dir, seen, count_only):
    rows, skipped, candidates = [], {}, 0
    for p in list_posts(s, pub_after, pub_before):
        pid = p.get("id")
        if pid in seen:
            continue
        seen.add(pid)
        open_t, close_t, resolve_t = times_of(p)
        if resolve_t is None or not keep(open_t, close_t, resolve_t):
            continue
        candidates += 1
        if count_only:
            continue
        detail = get(s, f"{BASE}/posts/{pid}/", {"with_cp": "true"})
        (raw_dir / f"{pid}.json").write_text(json.dumps(detail, indent=1))
        row, why = extract(detail, lead_days, min_forecasters)
        time.sleep(1.0)
        if row is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        o2, c2, r2 = times_of(detail)
        if not keep(o2, c2, r2):          # re-check with the detail record
            continue
        row["group"] = group
        rows.append(row)
        print(f"  [{group}] {len(rows):2d}/{n_want} id={pid} cp={row['cp_prob']:.2f} "
              f"res={row['resolution']} h={row['horizon_days']}d {row['title'][:55]}",
              file=sys.stderr)
        if len(rows) >= n_want:
            break
    print(f"  {group}: {candidates} candidates passed the date filters"
          + (f", skipped={skipped}" if skipped else ""), file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-start", required=True,
                    help="grouping boundary YYYY-MM-DD (our conservative choice, e.g. 2026-07-01 "
                         "for a model whose documented cutoff month is 2026-06)")
    ap.add_argument("--cutoff-label", default="",
                    help="the model's OFFICIAL cutoff as documented, stored verbatim in meta.json")
    ap.add_argument("--n-per-group", type=int, default=15)
    ap.add_argument("--lead-days", type=int, default=7)
    ap.add_argument("--pre-margin-days", type=int, default=120,
                    help="buffer between the end of the pre window and the boundary")
    ap.add_argument("--pre-window-days", type=int, default=365)
    ap.add_argument("--min-forecasters", type=int, default=10)
    ap.add_argument("--allow-closed-before-boundary", action="store_true",
                    help="post group: do NOT require the question to still be open after the boundary")
    ap.add_argument("--count-only", action="store_true",
                    help="only count questions passing the date filters; no detail fetches")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    boundary = datetime.strptime(a.post_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pre_end = boundary - timedelta(days=a.pre_margin_days)
    pre_start = pre_end - timedelta(days=a.pre_window_days)
    out = Path(a.out); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    s = session(); seen = set()

    def keep_post(o, c, r):
        if r <= boundary:
            return False
        return a.allow_closed_before_boundary or (c is not None and c > boundary)

    def keep_pre(o, c, r):
        return pre_start <= r <= pre_end

    print(f"POST: resolved after {boundary.date()}"
          + ("" if a.allow_closed_before_boundary else " and still open after it"), file=sys.stderr)
    post = collect(s, "post", boundary - timedelta(days=400), None, keep_post,
                   a.n_per_group, a.lead_days, a.min_forecasters, raw, seen, a.count_only)
    print(f"PRE: resolved in [{pre_start.date()}, {pre_end.date()}]", file=sys.stderr)
    pre = collect(s, "pre", pre_start - timedelta(days=400), pre_end, keep_pre,
                  a.n_per_group, a.lead_days, a.min_forecasters, raw, seen, a.count_only)

    meta = {"generated_at": datetime.now(timezone.utc).isoformat(),
            "model_cutoff_label": a.cutoff_label, "post_start_boundary": boundary.date().isoformat(),
            "post_requires_open_after_boundary": not a.allow_closed_before_boundary,
            "pre_window": [pre_start.date().isoformat(), pre_end.date().isoformat()],
            "lead_days": a.lead_days, "min_forecasters": a.min_forecasters,
            "n_post": len(post), "n_pre": len(pre)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if a.count_only:
        print("count-only run; see candidate counts above", file=sys.stderr)
        return

    rows = post + pre
    if not rows:
        sys.exit("no questions collected; inspect data/raw/*.json and the skip reasons above")
    with open(out / "questions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out / "prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "group", "title", "description",
                    "resolution_criteria", "fine_print", "open_time", "close_time")}) + "\n")
    for g, sub in (("post", post), ("pre", pre)):
        if sub:
            base = sum(r["resolution"] for r in sub) / len(sub)
            cpb = sum((r["cp_prob"] - r["resolution"]) ** 2 for r in sub) / len(sub)
            hz = sum(r["horizon_days"] for r in sub) / len(sub)
            print(f"{g}: n={len(sub)} base_rate(yes)={base:.2f} community_brier={cpb:.3f} "
                  f"mean_horizon={hz:.0f}d", file=sys.stderr)
    print(f"wrote {out/'questions.csv'}, {out/'prompts.jsonl'}, {out/'meta.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
