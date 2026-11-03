#!/usr/bin/env python3
"""
Score model forecasts vs the market, pre vs post cutoff.

Inputs:
  data/questions.csv      from scripts/pull_polymarket.py (has cp_prob, resolution, group; cp_open_prob if present)
  forecasts.csv           your model's output: columns  id,model_prob   (P(yes) in [0,1])

Reports per group: Brier for model, market (lead point), market near creation (if the column exists)
and constant 0.5; the model-minus-market gap; a bootstrap CI and a permutation p-value on the
between-group contrast (post gap minus pre gap). --table writes a per-question CSV.
--robustness reports timestamp/availability diagnostics and opening-price sensitivity checks;
--pilot and --horizon-split report optional exploratory subsets, without rerunning forecasts.

Usage:
  python scripts/score.py --questions data_v2/questions.csv --forecasts forecasts/forecasts_v2_claude-opus-5-5.csv \
                  --table data_v2/results_claude-opus-5-5.csv --flags data_v2/leak_claude-opus-5-5.csv
"""
import argparse, csv, random, statistics as st
from datetime import datetime

def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys)

def did(pre, post):  # positive = model has a larger relative loss in the post group
    return st.mean(post) - st.mean(pre)

def ci_and_perm(gaps, boot, perm, seed=0):
    """Resample paired per-question losses separately within pre and post.

    The percentile interval uses sorted draws at floor(.025*boot) and floor(.975*boot).
    A single seeded RNG generates bootstrap draws, then shuffles pooled question gaps for
    the two-sided label-permutation test, with the (hits+1)/(perm+1) correction. This is
    question-level uncertainty, not event clustering or repeated-generation uncertainty.
    """
    if boot < 1 or perm < 0:
        raise ValueError("boot must be positive and perm must be nonnegative")
    if not gaps.get("pre") or not gaps.get("post"):
        raise ValueError("both groups must contain at least one paired gap")
    obs = did(gaps["pre"], gaps["post"])
    rng = random.Random(seed); boots = []
    for _ in range(boot):
        boots.append(did(rng.choices(gaps["pre"], k=len(gaps["pre"])),
                         rng.choices(gaps["post"], k=len(gaps["post"]))))
    boots.sort()
    lo, hi = boots[int(0.025 * boot)], boots[int(0.975 * boot)]
    pooled = gaps["pre"] + gaps["post"]; n_pre = len(gaps["pre"]); hits = 0
    for _ in range(perm):
        rng.shuffle(pooled)
        if abs(did(pooled[:n_pre], pooled[n_pre:])) >= abs(obs):
            hits += 1
    return obs, lo, hi, (hits + 1) / (perm + 1)


def days_between(later, earlier):
    """Return an observed timestamp difference; do not impute missing timestamps."""
    if not later or not earlier:
        return None
    return (datetime.fromisoformat(later.replace("Z", "+00:00")) -
            datetime.fromisoformat(earlier.replace("Z", "+00:00"))).total_seconds() / 86400


def paired_gaps(rows, baseline):
    """Keep a model forecast and its chosen market baseline together, omitting absent prices."""
    return {g: [(r["m"] - r["y"]) ** 2 - (r[baseline] - r["y"]) ** 2
                for r in rows if r["group"] == g and r[baseline] is not None]
            for g in ("pre", "post")}


def report_subset(label, rows, baseline, boot, perm, seed):
    """Describe the actual paired subset and its exploratory between-group contrast."""
    print(f"\n{label}:")
    gaps = paired_gaps(rows, baseline)
    for g in ("pre", "post"):
        sub = [r for r in rows if r["group"] == g and r[baseline] is not None]
        if not sub:
            print(f"  {g:4s} n= 0")
            continue
        y = [r["y"] for r in sub]
        print(f"  {g:4s} n={len(sub):2d}  brier model={brier([r['m'] for r in sub], y):.3f}  "
              f"market={brier([r[baseline] for r in sub], y):.3f}  model-market={st.mean(gaps[g]):+.3f}")
    if gaps["pre"] and gaps["post"]:
        obs, lo, hi, p = ci_and_perm(gaps, boot, perm, seed)
        print(f"  post gap - pre gap = {obs:+.3f}   95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]   "
              f"permutation p = {p:.6f}")
    else:
        print("  contrast unavailable: one group has no paired observations")


def report_timing(rows):
    print("\nObserved baseline timing (days; matched forecasts only):")
    for g in ("pre", "post"):
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        available = [r for r in sub if r["c0"] is not None]
        lags = [r["open_lag"] for r in available if r["open_lag"] is not None]
        print(f"  {g:4s} near-creation n={len(available)} missing={len(sub)-len(available)}", end="")
        if lags:
            print(f"  timestamp-known={len(lags)}  lag median={st.median(lags):.3f}d "
                  f"max={max(lags):.3f}d  >1d={sum(x > 1 for x in lags)}")
        else:
            print("  lag unavailable")
        advances = [r["market_advance"] for r in sub if r["market_advance"] is not None]
        ages = [r["price_age"] for r in sub if r["price_age"] is not None]
        if advances:
            print(f"       lead-price date minus creation: mean={st.mean(advances):.3f}d (n={len(advances)})")
        if ages:
            print(f"       resolution minus lead-price date: mean={st.mean(ages):.3f}d "
                  f"range=[{min(ages):.3f}, {max(ages):.3f}]d (n={len(ages)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.csv")
    ap.add_argument("--forecasts", default="forecasts.csv")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--perm", type=int, default=20000, help="label-permutation draws (two-sided)")
    ap.add_argument("--seed", type=int, default=0, help="resampling seed, reset for each contrast")
    ap.add_argument("--robustness", action="store_true", help="report timing, lead-price results on the "
                    "opening-price-available subset, and opening prices within 24 hours")
    ap.add_argument("--pilot", default="", help="earlier questions CSV: report retained and newly added subsets")
    ap.add_argument("--horizon-split", type=float, default=None, metavar="DAYS",
                    help="report two horizon strata at this threshold")
    ap.add_argument("--table", default="", help="write a per-question CSV here")
    ap.add_argument("--flags", default="", help="scripts/leak_probe.py CSV: also report flagged/unflagged splits and the "
                                              "diff-in-diff on unflagged questions only")
    a = ap.parse_args()
    if a.boot < 1 or a.perm < 0:
        ap.error("--boot must be positive and --perm must be nonnegative")

    qs = {int(r["id"]): r for r in csv.DictReader(open(a.questions))}
    fc = {int(r["id"]): float(r["model_prob"]) for r in csv.DictReader(open(a.forecasts))}
    rows = [dict(id=i, group=qs[i]["group"], title=qs[i]["title"], m=fc[i], c=float(qs[i]["cp_prob"]),
                 c0=float(qs[i]["cp_open_prob"]) if qs[i].get("cp_open_prob") else None,
                 open_lag=days_between(qs[i].get("cp_open_time"), qs[i].get("open_time")),
                 market_advance=days_between(qs[i].get("cp_time"), qs[i].get("open_time")),
                 price_age=days_between(qs[i].get("resolve_time"), qs[i].get("cp_time")),
                 y=int(qs[i]["resolution"]), hz=float(qs[i].get("horizon_days") or 0))
            for i in qs if i in fc]
    missing = set(qs) - set(fc)
    if missing:
        print(f"warning: {len(missing)} questions have no model forecast: {sorted(missing)}")

    gaps, gaps0 = {}, {}
    for g in ("pre", "post"):
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        m = [r["m"] for r in sub]; c = [r["c"] for r in sub]; y = [r["y"] for r in sub]
        bm, bc, b5 = brier(m, y), brier(c, y), brier([0.5] * len(y), y)
        gaps[g] = [(p - yy) ** 2 - (q - yy) ** 2 for p, q, yy in zip(m, c, y)]
        wins = sum(1 for d in gaps[g] if d < 0)
        hz = st.mean(r["hz"] for r in sub); base = st.mean(y)
        extreme = sum(1 for p in m if p < 0.1 or p > 0.9)
        print(f"{g:4s} n={len(sub):2d}  brier model={bm:.3f}  market={bc:.3f}  const0.5={b5:.3f}  "
              f"model-market={bm-bc:+.3f}  beats market {wins}/{len(sub)}  "
              f"model<0.1|>0.9: {extreme}/{len(sub)}  yes-rate={base:.2f}  mean_horizon={hz:.0f}d")
        sub0 = [r for r in sub if r["c0"] is not None]
        if sub0:
            m0 = [r["m"] for r in sub0]; c0 = [r["c0"] for r in sub0]; y0 = [r["y"] for r in sub0]
            gaps0[g] = [(p - yy) ** 2 - (q - yy) ** 2 for p, q, yy in zip(m0, c0, y0)]
            print(f"     market near creation (n={len(sub0)}): brier model={brier(m0, y0):.3f}  "
                  f"market@open={brier(c0, y0):.3f}  model-market@open={brier(m0, y0)-brier(c0, y0):+.3f}")

    if "pre" in gaps and "post" in gaps:
        obs, lo, hi, p = ci_and_perm(gaps, a.boot, a.perm, a.seed)
        print(f"\ndiff-in-diff vs market at lead point (post gap - pre gap) = {obs:+.3f}   "
              f"95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]   permutation p = {p:.6f}")
        if "pre" in gaps0 and "post" in gaps0:
            obs0, lo0, hi0, p0 = ci_and_perm(gaps0, a.boot, a.perm, a.seed)
            print(f"diff-in-diff vs market near creation           = {obs0:+.3f}   "
                  f"95% bootstrap CI [{lo0:+.3f}, {hi0:+.3f}]   permutation p = {p0:.6f}")
        print("  Positive contrast = a larger model-minus-market loss gap in post; no causal effect is identified.")
        print("  A CI covering zero indicates imprecision, not evidence that the contrast is absent.")

    if a.flags and "pre" in gaps and "post" in gaps:
        flag = {int(r["id"]): r["knows_outcome"] == "1" for r in csv.DictReader(open(a.flags)) if r.get("run", "0") == "0"}
        print(f"\nsplit by automated later-information flag ({a.flags}):")
        sub_gaps = {}
        for g in ("pre", "post"):
            for label, want in (("flagged", True), ("unflagged", False)):
                sub = [r for r in rows if r["group"] == g and flag.get(r["id"]) is want]
                if not sub:
                    print(f"  {g:4s} {label:9s} n= 0"); continue
                m = [r["m"] for r in sub]; c = [r["c"] for r in sub]; y = [r["y"] for r in sub]
                gg = [(p - yy) ** 2 - (q - yy) ** 2 for p, q, yy in zip(m, c, y)]
                sub_gaps[(g, label)] = gg
                print(f"  {g:4s} {label:9s} n={len(sub):2d}  brier model={brier(m, y):.3f}  market={brier(c, y):.3f}  "
                      f"model-market={st.mean(gg):+.3f}")
        if ("pre", "unflagged") in sub_gaps and ("post", "unflagged") in sub_gaps:
            obs, lo, hi, p = ci_and_perm({"pre": sub_gaps[("pre", "unflagged")], "post": sub_gaps[("post", "unflagged")]},
                                         a.boot, a.perm, a.seed)
            print(f"  diff-in-diff on unflagged questions only = {obs:+.3f}   95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]   "
                  f"permutation p = {p:.6f}")

    if a.robustness:
        report_timing(rows)
        available = [r for r in rows if r["c0"] is not None]
        report_subset("Lead baseline on opening-price-available questions", available, "c", a.boot, a.perm, a.seed)
        within_day = [r for r in available if r["open_lag"] is not None and 0 <= r["open_lag"] <= 1]
        report_subset("Near-creation baseline restricted to lag <=24h", within_day, "c0", a.boot, a.perm, a.seed)

    if a.pilot:
        with open(a.pilot) as f:
            pilot_ids = {int(r["id"]) for r in csv.DictReader(f)}
        report_subset("Retained pilot questions (lead baseline)", [r for r in rows if r["id"] in pilot_ids],
                      "c", a.boot, a.perm, a.seed)
        report_subset("Newly added questions (lead baseline)", [r for r in rows if r["id"] not in pilot_ids],
                      "c", a.boot, a.perm, a.seed)

    if a.horizon_split is not None:
        for label, sub in ((f"Horizon <= {a.horizon_split:g}d", [r for r in rows if r["hz"] <= a.horizon_split]),
                           (f"Horizon > {a.horizon_split:g}d", [r for r in rows if r["hz"] > a.horizon_split])):
            report_subset(label + " (lead baseline)", sub, "c", a.boot, a.perm, a.seed)

    print(f"\nMethods: {a.boot} draws of paired question-level gaps, resampled separately within pre/post; "
          f"percentile CI at sorted indices floor(.025*B), floor(.975*B); seed={a.seed} per contrast.")
    print(f"  {a.perm} two-sided pooled-label permutations after bootstrap draws, plus-one p-value correction.")
    print("  Conditional on this sample and one forecast per question; no event clustering or multiplicity correction.")
    print("  Permutation inference assumes exchangeability across groups; all contrasts are exploratory.")

    if a.table:
        with open(a.table, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "id", "title", "horizon_days", "resolution", "market_p", "market_open_p",
                        "model_p", "market_sq", "model_sq", "model_minus_market"])
            for r in sorted(rows, key=lambda r: (r["group"] != "pre", r["id"])):
                msq, csq = (r["m"] - r["y"]) ** 2, (r["c"] - r["y"]) ** 2
                w.writerow([r["group"], r["id"], r["title"], r["hz"], r["y"], r["c"],
                            "" if r["c0"] is None else r["c0"], r["m"], f"{csq:.4f}", f"{msq:.4f}", f"{msq-csq:+.4f}"])
        print(f"wrote {a.table}")

if __name__ == "__main__":
    main()
