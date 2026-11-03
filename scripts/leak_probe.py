#!/usr/bin/env python3
"""
Transcript-evidence probe: a second model reads each forecast's reasoning and flags language
that states or appears to rely on an outcome or an event after the pinned "today". The legacy
column name ``knows_outcome`` denotes this combined textual flag, not verified internal
knowledge, successful recall, or a causal measure of training-data contamination. A negative
flag means no such evidence was detected in the short explanation, not that leakage is absent.

Input : data/model_raw/<model>/<id>_r<k>.json   (from scripts/forecast.py)
Output: <out> CSV: id, group, run, knows_outcome, confidence, evidence
The judge sees the title, pinned date and reasoning text only (not the probability, full
forecast input, or resolved outcome). It does not verify claims against external sources or
distinguish recalled outcomes from intermediate events, prompt-supplied facts, or false claims.
The historical judge prompt and schema below are retained to match the existing labels.

Usage:
  python scripts/leak_probe.py --model claude-opus-5-5 --questions data_v2/questions.csv --out data_v2/leak_claude-opus-5-5.csv
"""
import argparse, csv, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

SCHEMA = {
    "type": "object",
    "properties": {
        "knows_outcome": {"type": "boolean",
                          "description": "True if the reasoning states or clearly relies on knowledge of how the question resolved, "
                                         "or of specific events after 'today'. False if it only forecasts, or merely says it does not know."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {"type": "string", "description": "The shortest quote from the reasoning that supports the verdict, or ''."},
    },
    "required": ["knows_outcome", "confidence", "evidence"],
    "additionalProperties": False,
}

SYSTEM = ("You audit forecasting transcripts for information leakage. A forecaster was told that 'today' is a given "
          "date and asked to forecast a question using only knowledge available on that date. You will see the question, "
          "the pinned date and the forecaster's reasoning. Decide whether the reasoning uses knowledge of the actual "
          "outcome or of specific events that happened after the pinned date. Hedged recollection counts as knowing "
          "ONLY when the thing recalled is the outcome itself or an event dated after the pinned date ('I recall "
          "that X won the June 24 primary' with June 24 after the pinned date). Recalling facts from before the "
          "pinned date ('as I recall, the April report showed...' with April before the pinned date) is ordinary "
          "forecasting and does NOT count, even if phrased as a recollection. A plain disclaimer ('I do not know "
          "how this resolved', 'I cannot know events after today') does NOT count. Ordinary forecasting from base "
          "rates, polls or trends known before the date does NOT count. Compare every recalled date against the "
          "pinned date before deciding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="forecaster whose transcripts are judged")
    ap.add_argument("--judge", default="claude-haiku-5-5")
    ap.add_argument("--questions", default="data/questions.csv")
    ap.add_argument("--raw-dir", default="data/model_raw")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="default data/leak_<model>.csv")
    a = ap.parse_args()
    out = Path(a.out or f"data/leak_{a.model}.csv")

    qs = {r["id"]: r for r in csv.DictReader(open(a.questions))}
    jobs = []
    for qid, q in qs.items():
        for k in range(a.runs):
            fp = Path(a.raw_dir) / a.model / f"{qid}_r{k}.json"
            if fp.exists():
                jobs.append((q, k, json.loads(fp.read_text())))
    print(f"{len(jobs)} transcripts to judge with {a.judge}", file=sys.stderr)
    client = anthropic.Anthropic(timeout=120, max_retries=3)

    def judge(q, k, rec):
        user = (f"Pinned date ('today'): {q['open_time'][:10]}\n\nQuestion: {q['title']}\n\n"
                f"Forecaster's reasoning:\n\"\"\"\n{rec.get('reasoning', '')}\n\"\"\"")
        # The judge deliberately does NOT see the forecaster's probability: a confident number must not
        # nudge the verdict, since the flag is later used to split the Brier scores.
        resp = client.messages.create(model=a.judge, max_tokens=600, system=SYSTEM,
                                      messages=[{"role": "user", "content": user}],
                                      output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        text = next((b.text for b in resp.content if b.type == "text"), "")
        d = json.loads(text)
        return [q["id"], q["group"], k, int(bool(d["knows_outcome"])), d["confidence"], d["evidence"]]

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(judge, q, k, rec): (q, k) for q, k, rec in jobs}
        for f in as_completed(futs):
            q, k = futs[f]
            try:
                rows.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"  id={q['id']} r{k} FAILED: {e}", file=sys.stderr)
    rows.sort(key=lambda r: (r[1] != "pre", int(r[0]), r[2]))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "group", "run", "knows_outcome", "confidence", "evidence"]); w.writerows(rows)
    for g in ("pre", "post"):
        sub = [r for r in rows if r[1] == g]
        if sub:
            print(f"{a.model} {g}: judged knows_outcome {sum(r[3] for r in sub)}/{len(sub)}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
