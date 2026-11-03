# Leakage study: pre- vs post-cutoff forecasting against a prediction market

Data source is **Polymarket** (public Gamma + CLOB APIs, no auth). The original Metaculus pipeline (`scripts/legacy/pull_metaculus.py`) is retained for reference; the project's earlier attempts encountered unavailable fields (see HANDOFF.md §6). Results are in [REPORT.md](REPORT.md). The current sample is 40 + 40 questions in `data_v2/`; the first 15 + 15 run is kept in `data/`.

**To reproduce the report from saved data, start with [REPRODUCE.md](REPRODUCE.md).** Its scoring commands need only Python's standard library, no credentials or network calls. They include near-creation timing, a 24-hour price-window check, matched-subset comparisons, retained/new questions, and flag subsets. Complete saved output: [Opus](analysis/score-opus.txt), [Haiku](analysis/score-haiku.txt). The result is an exploratory association, not an identified memory effect. The sections below also document how to collect data and query models for a new experiment.

## Project structure

```text
README.md                 Setup and workflow
REPORT.md                 Research report
REPRODUCE.md              Saved-data reproduction and audit details
HANDOFF.md                Historical project notes
scripts/                  Data collection, forecasting, scoring, and probing
  legacy/                 Earlier Metaculus collection pipeline
forecasts/                Saved pilot and 80-question forecast CSVs
analysis/                 Saved scoring output
data/                    Pilot data and shared model-response cache
data_v2/                 80-question dataset and historical judge labels
requirements.txt          Dependencies for data collection and model calls
```

Run all commands from the project root. The scoring script requires only the Python standard library. `data/model_raw/` remains the shared cache for both samples.

## 0. Setup (once)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (METACULUS_TOKEN only for the legacy script)
set -a; . ./.env; set +a
```

## 1. Pull data
```bash
python scripts/pull_polymarket.py --post-start 2026-07-01 --cutoff-label "2026-06 training data cutoff (Anthropic docs)" --count-only
python scripts/pull_polymarket.py --post-start 2026-07-01 --cutoff-label "2026-06 training data cutoff (Anthropic docs)" --n-per-group 15
# the 40-per-group sample (data_v2/): slice the listing by week, keep the earlier sample's markets, draw 44, then trim to 40 (see below)
python scripts/pull_polymarket.py --post-start 2026-07-01 --cutoff-label "2026-06 training data cutoff (Anthropic docs)" --n-per-group 44 --slice-days 7 --max-pages 11 --keep data/questions.csv --out data_v2
```
Gamma's listing stops at offset ~2000 (HTTP 422), so without `--slice-days` only the ~2000 highest-volume markets per group are ever seen; a slice that is still full after `--max-pages` pages prints a WARNING (lower `--slice-days` or raise `--max-pages`). `--keep` places the ids of an earlier sample first (after the shuffle) so the new sample *extends* it: this is extension sampling that prefers the earlier questions, not a uniform draw from the new pool. `--max-per-topic` relaxes the per-topic cap. The published `data_v2/` sample was then trimmed deterministically: drop id 556167 (Ryder Cup, golf; slipped past the keyword filter) and keep the first 40 per group in the file's (shuffle) order; `data_v2/meta.json` records this. Note: `data_v2/` was pulled with an earlier version of the slicing that left a one-day gap between slices and shared the de-duplication set across groups (both fixed since); its candidate pool was therefore slightly incomplete.
`--post-start` is OUR grouping boundary (first day after the documented cutoff month); the official cutoff goes verbatim into `--cutoff-label` and is saved to `data/meta.json`. Run `--count-only` first to see how many markets pass the filters.

Outputs in `data/`:
- `questions.csv` – full table incl. `cp_prob` (YES price `--lead-days` before resolution), `cp_open_prob` (first daily price point at or after `createdAt`, at most 7 days later, blank otherwise; in `data_v2/` the median lag is under a day but about a quarter of the points are more than a day after creation) and `resolution`. **Do not show this to the model.**
- `prompts.jsonl` – market question + rules only (no resolution, no price). Feed this to the model.
- `raw/*.json` – market record + daily price history.

Groups:
- `post`: resolved after the boundary AND the market's own deadline (`endDate`) is more than 3 days after it (so "by June 30?" markets resolving on July 1 are excluded).
- `pre`: resolved in a one-year window ending `--pre-margin-days` (120) before the boundary: possible training exposure, not guaranteed.

Market filters: binary Yes/No, resolved YES or NO, volume >= $20k, horizon >= 14 days, not sports / esports / crypto-price / commodity-price (keyword filter; the listing API carries no tags), one market per event, at most 2 markets per capitalised title word per group, YES price at the lead point in [0.03, 0.97] (drops markets the market had already called). Markets are drawn at random (`--seed`) from the candidates, subject to the one-per-event and per-topic caps and, with `--keep`, after the earlier sample's markets.

## 2. Query the model
```bash
python scripts/forecast.py --model claude-opus-5-5
python scripts/forecast.py --model claude-haiku-5-5
# 40-per-group sample: responses are cached per question id in data/model_raw/, so questions kept from the earlier sample are not re-queried.
# The cache key is the question id only (not the prompt text or --effort): pass --overwrite after changing either.
python scripts/forecast.py --model claude-opus-5-5 --prompts data_v2/prompts.jsonl --out forecasts/forecasts_v2_claude-opus-5-5.csv --usage data_v2/usage_claude-opus-5-5.csv
```
`scripts/forecast.py` (Anthropic SDK, no tools, system prompt "today is <open_time>", structured JSON output) writes `forecasts/forecasts_<model>.csv` (`id,model_prob`), per-run responses in `data/model_raw/<model>/`, and token usage + estimated cost in `data/usage_<model>.csv`. `--runs 3` averages three samples per question; `--effort` sets thinking depth (default `high`). Refusals and parse failures are retried once, then left missing (no model fallback, so every forecast comes from the named model).

## 3. Score
```bash
python scripts/score.py --forecasts forecasts/forecasts_claude-opus-5-5.csv
python scripts/score.py --questions data_v2/questions.csv --forecasts forecasts/forecasts_v2_claude-opus-5-5.csv --table data_v2/results_claude-opus-5-5.csv
# full saved-data sensitivity analysis, without overwriting result tables
python scripts/score.py --questions data_v2/questions.csv --forecasts forecasts/forecasts_v2_claude-opus-5-5.csv --flags data_v2/leak_claude-opus-5-5.csv --robustness --pilot data/questions.csv --horizon-split 60
```
Prints Brier (model / market / market near creation / constant 0.5) per group, a bootstrap CI and a two-sided permutation p on the contrast: (model − market) in post minus the same gap in pre. Bootstrap resamples paired question-level loss differences within each group; label permutations require exchangeability. Defaults: `--boot 5000 --perm 20000 --seed 0`. `--robustness` adds price timing, matched-subset and 24-hour-window checks; `--pilot PATH` splits retained/new questions; `--horizon-split 60` adds coarse horizon strata. These are exploratory, without event clustering or multiple-comparison adjustment. `--table` writes a per-question CSV.

## 4. Leakage probe
```bash
python scripts/leak_probe.py --model claude-opus-5-5 --questions data_v2/questions.csv --out data_v2/leak_claude-opus-5-5.csv
```
A judge model (default `claude-haiku-5-5`) reads the title, pinned date and forecast's reasoning text (not the probability or full supplied rules) and flags apparent outcome or post-date information. Output: `id,group,run,knows_outcome,confidence,evidence`. The legacy `knows_outcome` name does not mean actual memory has been validated. Positive flags can combine different mechanisms, and negative flags do not certify absence of later information. No blinded human validation has been completed; selected AI-assisted checks and their limits are in [REPRODUCE.md](REPRODUCE.md). The previously reported 78/80 agreement involved a changed judge prompt, not a clean test of stochastic reliability. `scripts/score.py --flags <this csv>` reports flagged/unflagged scores and the selected unflagged-only contrast, which does not estimate memory's causal contribution.

## If the API shape differs
Polymarket: `scripts/pull_polymarket.py` reads `closedTime` (resolution time), `endDate` (scheduled deadline), `outcomePrices` (`["1","0"]` = YES) from Gamma `/markets`, and daily prices from CLOB `prices-history?interval=max&fidelity=1440`. Skip reasons printed to stderr say which filter removed a market.

Metaculus (legacy): `scripts/legacy/pull_metaculus.py` needs an Unrestricted or Bot-Benchmarking API tier (request via the Metaculus Data Needs Form); with a default token every `resolution` and aggregation field comes back `null`.
