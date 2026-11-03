# Reproducing the saved-data analysis

This document accompanies [REPORT.md](REPORT.md). The November 2, 2026 revision recomputes existing forecasts and adds sensitivity analyses and limited AI-assisted transcript checks. It does not redraw questions, regenerate forecasts, rerun the judge, or complete human validation.

## 1. Recompute without network access or model charges

Run from this project directory. `scripts/score.py` uses the Python standard library; the existing `.venv/bin/python` also works. These commands read the saved experiment and print all reported comparisons:

```bash
python3 scripts/score.py --questions data_v2/questions.csv \
  --forecasts forecasts/forecasts_v2_claude-opus-5-5.csv \
  --flags data_v2/leak_claude-opus-5-5.csv \
  --robustness --pilot data/questions.csv --horizon-split 60

python3 scripts/score.py --questions data_v2/questions.csv \
  --forecasts forecasts/forecasts_v2_claude-haiku-5-5.csv \
  --flags data_v2/leak_claude-haiku-5-5.csv \
  --robustness --pilot data/questions.csv --horizon-split 60
```

Saved outputs from these settings: [Opus](analysis/score-opus.txt), [Haiku](analysis/score-haiku.txt). Defaults are `--boot 5000 --perm 20000 --seed 0`. `--table PATH` optionally writes per-question scores; choose a new path to retain historical tables. Existing forecasts, probabilities, resolutions and judge labels were not changed for this revision.

Files used:

- `data_v2/questions.csv`: 80 market records, including outcomes and both market baselines. This is scoring data, not model input.
- `data_v2/prompts.jsonl`: title, supplied rules and dates shown to the forecaster.
- `forecasts/forecasts_v2_claude-{opus,haiku}-5-5.csv`: one saved forecast per question for each model.
- `data/model_raw/claude-{opus,haiku}-5-5/<id>_r0.json`: saved probabilities, explanations, returned model identifiers and usage; these also store the 53 added questions despite the `data/` directory name.
- `data_v2/leak_claude-{opus,haiku}-5-5.csv`: historical automated flags, confidence and evidence quotes. The analysis uses run 0.
- `data/questions.csv`: the original 30-question pilot, used only to identify retained/new questions.
- `data_v2/raw/<id>.json`: fetched market records and daily price histories. These are collection-time records, not creation-time text snapshots.

## 2. Statistical definition and exact implementation

For each question, compute model squared error minus market squared error. Report the mean in each group and the later-minus-earlier contrast. The bootstrap samples these paired differences with replacement, separately within each group at its observed size. It never independently resamples model and market scores.

For each contrast, initialize Python `random.Random(0)`, draw 5,000 bootstrap replicates, sort them, and use zero-based indices 125 and 4875 as the 95% percentile endpoints. The same RNG then generates 20,000 permutations of the pooled paired differences while retaining the original group sizes. The two-sided p-value is `(number of |permuted contrast| >= |observed contrast| + 1) / 20001`. The seed resets for each comparison. Changing the bootstrap count also changes the subsequent permutation sequence. Reported table cells are independently rounded; subtracting rounded cells can differ from the rounded paired difference.

Resampling units are questions, not repeated model runs or underlying-event clusters. All 80 first platform event IDs in the saved raw records are distinct. This does not establish independence: questions 601826 and 601920, for example, both concern the 2026 Brazilian presidential election. Related events can make question-level intervals too narrow. The observational groups need not satisfy permutation exchangeability. There is no multiple-comparison correction; subgroup analyses are exploratory and conditional on the selected sample.

The intervals do not quantify collector bias, selection after the pilot, text-label error, unavailable historical inputs, or generation-to-generation variability. An interval including zero is not evidence of equivalence.

## 3. Market timestamps, missingness and sensitivity checks

The lead baseline selects the last available daily price at or before settlement minus seven days. The collector rejects missing prices and prices whose age relative to that target has integer `.days > 3`; this implementation allows lags below four days, rather than enforcing an exact 72-hour maximum. In this saved sample, prices precede settlement by 7.009–7.921 days in `pre` and 7.005–8.591 days in `post`. The mean market timestamp is 52.449/125.649 days after creation. These are actual timestamps; the approximate 53/126-day values in the report are mean horizon minus seven days. The model is only given the calendar date, not the exact creation timestamp.

The near-creation baseline selects the earliest daily observation at or after creation, within seven days; if none exists, the value stays missing. It is not necessarily a transaction at creation or a liquid quoted price. No volume-weighting, interpolation, minimum opening-liquidity filter, or carry-forward imputation is applied.

- Availability: earlier 40/40, later 38/40. Missing later IDs: **601826 and 2166702**. Both remain in the 80-question lead-price analysis.
- Median lag after creation: **0.296/0.323 days**; maximum: **2.310/3.973 days**; lag above one day: **10/11**.
- On the matched 40/38 near-creation sample, Opus Brier is **0.109/0.302**, Haiku **0.227/0.336**, market **0.310/0.230**. Opus contrast **+0.272 [+0.156, +0.395]**; Haiku **+0.189 [+0.070, +0.313]**.
- Using lead prices on those same 78 questions gives Opus **+0.180 [+0.068, +0.300]** and Haiku **+0.096 [−0.053, +0.245]**. This separates the change in benchmark from dropping two questions.
- Restricting the near-creation baseline to an observed lag from zero through 24 hours leaves **30/27** questions. Opus **+0.352 [+0.215, +0.499]**; Haiku **+0.220 [+0.073, +0.376]**.

The stricter window also changes which questions remain. Its larger estimate does not identify a timing effect. Even an exactly matched timestamp would not supply the model with background published after its training cutoff but before that timestamp.

## 4. Pilot, horizon and flag subsets

The full CLI outputs linked above contain sample sizes, both Brier scores, paired differences and intervals for every subset. Key checks:

- Retained pilot: **14 pre/13 post**. Opus **+0.096 [−0.124, +0.312]**; Haiku **−0.039 [−0.319, +0.228]**.
- Newly added: **26/27**. Opus **+0.216 [+0.075, +0.370]**; Haiku **+0.158 [−0.009, +0.342]**. These are not an independent validation sample because the expansion shares the pilot-informed selection process.
- Horizon at most 60 days: **27/19**. Opus **+0.212 [+0.059, +0.381]**; Haiku **+0.217 [+0.017, +0.429]**.
- Horizon over 60 days: **13/21**. Opus **+0.132 [−0.054, +0.318]**; Haiku **−0.074 [−0.318, +0.165]**. Coarse strata do not equalize horizons or topics within strata.
- Earlier Opus flagged/unflagged subsets: **30/10**. Model Brier **0.073/0.214**, market **0.192/0.162**, model-minus-market **−0.118/+0.052**. Later unflagged n=40, model **0.309**, market **0.208**, difference **+0.101**.
- Opus unflagged-only contrast: **+0.049 [−0.079, +0.175]**. This compares selected subsets and cannot estimate the share of the full-sample contrast attributable to memory.

## 5. What the probe labels, and what was checked

`scripts/leak_probe.py` gives the judge the question title, date and explanation. It withholds the probability but also omits the full input rules, resolution and independent evidence about publication dates. The legacy field `knows_outcome` combines outcome assertions and later-event assertions; its name must not be interpreted as ground truth about internal knowledge. The historical judge prompt, schema and saved labels are unchanged.

A previous report described agreement on 78/80 transcripts after changing the judge to withhold probability. The two runs are not retained here as a versioned comparison, and the prompt changed. This is not a clean repeated-run reliability estimate and is not used as validation in the revised report.

The following are **purposefully selected AI-assisted case checks**, not a random audit, blinded human review, precision/recall estimate, or comprehensive fact verification. Only the FOMC outcome below was corroborated against an external primary source in this revision. For other cases, quotations are claims made by the model, not independently established facts.

- **528974, earlier, flagged, assigned date 2025-03-19: corroborated outcome assertion.** [Explanation](data/model_raw/claude-opus-5-5/528974_r0.json): “at the July 29-30, 2025 meeting the FOMC held rates at 4.25-4.50%, with Governors Waller and Bowman dissenting in favor of a cut.” The [Federal Reserve's July 30 statement](https://www.federalreserve.gov/monetarypolicy/monetary20250730a.htm) confirms the decision and dissenters. Both saved pilot and expanded inputs give the meeting schedule and resolution rules, but not these outcome details. This is evidence of a retrospective assertion despite the date instruction, not direct observation of training memory.
- **690635, earlier, flagged, assigned date 2025-11-18: intermediate-event claim.** [Explanation](data/model_raw/claude-opus-5-5/690635_r0.json) and judge evidence state that the Epstein Files Transparency Act became law around November 19, 2025. That claim is absent from input and is about an enabling event, not the target question's outcome concerning Stephen Colbert. The explanation also discusses December releases. Its factual accuracy is not verified here; it illustrates why the flag is broader than final-answer recall.
- **540159, earlier, unflagged, assigned date 2025-04-29: no obvious dated violation in the selected passage.** [Explanation](data/model_raw/claude-opus-5-5/540159_r0.json) invokes an April 28 election before forecasting parliamentary timing. The cited date precedes the assigned date. That observation does not certify the response as uncontaminated or verify its factual claims.
- **618557, earlier, unflagged, assigned date 2025-09-30: potential missed temporal claim.** [Explanation](data/model_raw/claude-opus-5-5/618557_r0.json) says “Russian claims about Siversk-area towns came later in the year.” This assertion is absent from the supplied input. Its retrospective wording warrants review, but identifying and dating the underlying event is necessary before calling it a verified false negative.
- **638492, earlier, flagged, assigned date 2025-10-17: ambiguous positive.** [Explanation](data/model_raw/claude-opus-5-5/638492_r0.json) vaguely recalls an edgeX token event being targeted for 2026. A future target date does not show that the announcement itself postdated October 17. The publication date needs checking.
- **601826 and 601920, later, unflagged, assigned date 2025-09-17: negative examples from the later group.** These were selected as the two smallest later-group IDs. Their [first](data/model_raw/claude-opus-5-5/601826_r0.json) and [second](data/model_raw/claude-opus-5-5/601920_r0.json) explanations discuss candidate prospects and historical election patterns without an explicit assertion of the 2026 result. The dates of all political and polling claims have not been independently established. They therefore remain unchecked negatives, not evidence of perfect detection or absence of contamination.
- **1280179, earlier, unflagged, assigned date 2026-01-27: input provenance example.** The [explanation](data/model_raw/claude-opus-5-5/1280179_r0.json) explicitly cites January 12 and January 25–26 examples supplied in the rules. Those dates precede the assigned date. This demonstrates that factual detail may originate in the prompt; it is not a demonstrated instance of future-information leakage.

An appropriate next audit would sample positive and negative labels in both groups, use independent reviewers and dated sources, and record the specific claim, event/publication date, whether it was in the input, accuracy, and unresolved ambiguity. `updatedAt > createdAt` alone does not prove that text was substantively changed. Creation-time snapshots are needed to investigate input leakage across the full sample.

## 6. Sampling provenance and model records

The recorded expansion prioritized pilot IDs after shuffling, applied event/topic caps, requested 44 markets per group, removed golf market 556167 (Ryder Cup), and retained the first 40 per group. See [meta.json](data_v2/meta.json). The earlier narrative records candidate counts of 356 later and 1,100 earlier, but complete candidate-pool snapshots are not saved; these counts cannot be reconstructed from the final 80 raw records alone.

The collector previously left one-day gaps between listing slices and shared a de-duplication set between groups. The repaired code uses contiguous windows and separate sets, but the published sample predates those repairs. Missing pilot IDs 2734313 and 532742 were attributed to the slice gaps; 5340071 was thought to have hit a page cap, without individual confirmation. Sports keyword rules were expanded during collection. Existing records do not establish a frozen pre-outcome protocol. Incomplete coverage can alter the composition and generalizability of the sample; a seeded shuffle does not fix it. A new sample should use a different output directory and save full candidate logs.

The saved 27 reused prompt records agree in all rendered fields between `data/prompts.jsonl` and `data_v2/prompts.jsonl`. Cached answers therefore show no detected input mismatch for this reuse. However, `scripts/forecast.py` caches by model directory, question ID and run number, not by prompt hash or effort. Future changes require a fresh cache or `--overwrite`.

The [official model overview](https://platform.claude.com/docs/en/models/overview), checked November 2, 2026, lists `claude-opus-5-5` and `claude-haiku-5-5` with June 2026 training-data cutoffs. Stored responses include returned model identifiers and usage, but not full serialized requests, prompt hashes or per-call timestamps. Collection dates in metadata and filesystem times cannot substitute for exact invocation dates. Future runs should save these fields and a dated documentation snapshot. The reported cutoff does not reveal exposure to any individual fact.

The prior cost estimates were $1.26 for Opus and $0.04 for Haiku across the cached 80-question results, not new costs incurred by this saved-data revision. Exact original-run billing is not independently audited here.
