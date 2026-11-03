#!/usr/bin/env python3
"""
Step 3 of the leakage study (HANDOFF.md §4.3): ask a Claude model for P(YES) on
every question in data/prompts.jsonl and write forecasts/forecasts_<model>.csv (id,model_prob).

What the model sees: ONLY the question title, description, resolution criteria and
fine print, plus a system prompt that pins "today" to the question's open_time.
No tools, no web search, no community probability, no resolution.

Usage:
  set -a; . ./.env; set +a            # ANTHROPIC_API_KEY
  python scripts/forecast.py --model claude-opus-5-5
  python scripts/forecast.py --model claude-haiku-5-5
  python scripts/forecast.py --model claude-opus-5-5 --runs 3      # average 3 samples per question
  python scripts/forecast.py --model claude-opus-5-5 --limit 2 --dry-run

Outputs:
  forecasts/forecasts_<model>.csv            id,model_prob            (mean over runs)
  data/model_raw/<model>/<id>_r<k>.json   full API response per run (reasoning, usage)
  data/usage_<model>.csv           per-run input/output tokens and estimated USD

Design notes:
  - No server-side refusal fallback on purpose: a fallback would substitute a
    different model's forecast. A refusal / parse failure is retried once, then
    recorded as missing (scripts/score.py reports missing ids).
  - Thinking is left at the model default (adaptive); depth via --effort.
"""
import argparse, csv, json, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

# USD per 1M tokens (input, output) - platform.claude.com/docs pricing, 2026-10.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string",
                      "description": "3-6 sentences of reasoning using only what you knew as of today."},
        # (no minimum/maximum: the structured-output schema rejects them; range is checked in ask())
        "probability": {"type": "number",
                        "description": "Probability that the question resolves YES, between 0 and 1."},
    },
    "required": ["reasoning", "probability"],
    "additionalProperties": False,
}


def system_prompt(q):
    today = q["open_time"][:10]
    return (
        f"You are an expert forecaster. Today is {today}. You have no internet access and no tools; "
        f"use only what you know as of today ({today}) and do not assume knowledge of anything that "
        f"happened after this date. If you believe you already know how this question actually "
        f"resolved, say so explicitly in your reasoning. Give a calibrated probability that the "
        f"question resolves YES, as a number strictly between 0 and 1."
    )


def user_prompt(q):
    parts = [f"# Question\n{q['title']}"]
    if q.get("description"):
        parts.append(f"# Background\n{q['description']}")
    if q.get("resolution_criteria"):
        parts.append(f"# Resolution criteria\n{q['resolution_criteria']}")
    if q.get("fine_print"):
        parts.append(f"# Fine print\n{q['fine_print']}")
    parts.append(f"The question opened on {q['open_time'][:10]} and closes for forecasting on "
                 f"{q['close_time'][:10]}.")
    parts.append('Respond with JSON: {"reasoning": string, "probability": number in (0,1)}.')
    return "\n\n".join(parts)


def ask(client, q, model, effort, max_tokens):
    kwargs = dict(model=model, max_tokens=max_tokens, system=system_prompt(q),
                  messages=[{"role": "user", "content": user_prompt(q)}],
                  output_config={"format": {"type": "json_schema", "schema": SCHEMA},
                                 **({"effort": effort} if effort else {})})
    resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {getattr(resp, 'stop_details', None)}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens before finishing")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    data = json.loads(text)             # output_config.format guarantees valid JSON
    p = float(data["probability"])
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"probability out of range: {p}")
    return p, data.get("reasoning", ""), resp


def cost_usd(model, usage):
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return usage.input_tokens / 1e6 * pin + usage.output_tokens / 1e6 * pout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. claude-opus-5-5 or claude-haiku-5-5")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--out", default=None, help="default forecasts/forecasts_<model>.csv")
    ap.add_argument("--usage", default=None, help="default data/usage_<model>.csv")
    ap.add_argument("--runs", type=int, default=1, help="samples per question, averaged")
    ap.add_argument("--effort", default="high", help="low|medium|high|xhigh|max; '' = model default")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="ignore existing raw responses")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, call nothing")
    a = ap.parse_args()

    out = Path(a.out or f"forecasts/forecasts_{a.model}.csv")
    raw_dir = Path("data/model_raw") / a.model
    usage_path = Path(a.usage or f"data/usage_{a.model}.csv")

    qs = [json.loads(l) for l in open(a.prompts) if l.strip()]
    if a.limit:
        qs = qs[:a.limit]
    if a.dry_run:
        for q in qs:
            print("=" * 80, f"\nid={q['id']} group={q.get('group')}\nSYSTEM: {system_prompt(q)}\nUSER:\n{user_prompt(q)}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(timeout=a.timeout, max_retries=3)
    if a.model not in PRICES:
        print(f"warning: no price entry for {a.model}; cost will show as 0", file=sys.stderr)

    jobs = []   # (question, run_index)
    for q in qs:
        for k in range(a.runs):
            if not a.overwrite and (raw_dir / f"{q['id']}_r{k}.json").exists():
                continue
            jobs.append((q, k))
    print(f"{len(qs)} questions x {a.runs} run(s); {len(jobs)} calls to make "
          f"(model={a.model}, effort={a.effort!r})", file=sys.stderr)

    def work(q, k):
        last = None
        for attempt in range(2):                       # one retry on refusal / parse error
            try:
                p, reasoning, resp = ask(client, q, a.model, a.effort or None, a.max_tokens)
                rec = {"id": q["id"], "run": k, "group": q.get("group"), "model": resp.model,
                       "probability": p, "reasoning": reasoning,
                       "stop_reason": resp.stop_reason,
                       "usage": resp.usage.model_dump(), "cost_usd": cost_usd(a.model, resp.usage)}
                (raw_dir / f"{q['id']}_r{k}.json").write_text(json.dumps(rec, indent=1))
                return q, k, rec, None
            except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
                last = e; time.sleep(5 * (attempt + 1))
            except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as e:
                last = e; time.sleep(2)
        return q, k, None, last

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, q, k) for q, k in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            q, k, rec, err = f.result()
            if err is not None:
                print(f"  [{i}/{len(jobs)}] id={q['id']} r{k} FAILED: {err}", file=sys.stderr)
                continue
            u = rec["usage"]
            print(f"  [{i}/{len(jobs)}] id={q['id']} r{k} {rec['group']:4s} p={rec['probability']:.3f} "
                  f"in={u['input_tokens']} out={u['output_tokens']} ${rec['cost_usd']:.4f}  "
                  f"{q['title'][:55]}", file=sys.stderr)

    # Aggregate everything on disk for this model (including earlier runs).
    per_q, usage_rows, total = {}, [], 0.0
    for q in qs:
        ps = []
        for k in range(a.runs):
            fp = raw_dir / f"{q['id']}_r{k}.json"
            if not fp.exists():
                continue
            rec = json.loads(fp.read_text())
            ps.append(rec["probability"])
            u = rec["usage"]; total += rec["cost_usd"]
            usage_rows.append([q["id"], k, u["input_tokens"], u["output_tokens"],
                               f"{rec['cost_usd']:.5f}"])
        if ps:
            per_q[q["id"]] = statistics.mean(ps)

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "model_prob"])
        for qid in sorted(per_q):
            w.writerow([qid, f"{per_q[qid]:.4f}"])
    with open(usage_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "run", "input_tokens", "output_tokens", "cost_usd"])
        w.writerows(usage_rows)

    n_in = sum(r[2] for r in usage_rows); n_out = sum(r[3] for r in usage_rows)
    print(f"\nwrote {out}: {len(per_q)}/{len(qs)} questions forecast in {time.time()-t0:.0f}s", file=sys.stderr)
    print(f"usage: {len(usage_rows)} calls, {n_in} input + {n_out} output tokens, "
          f"est. ${total:.3f} ({a.model})  -> {usage_path}", file=sys.stderr)
    missing = [q["id"] for q in qs if q["id"] not in per_q]
    if missing:
        print(f"missing forecasts for ids: {missing} (re-run to retry)", file=sys.stderr)


if __name__ == "__main__":
    main()
