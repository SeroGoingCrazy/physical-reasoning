# Forecasting scores and later-information use: an exploratory Polymarket study

*Physical Reasoning research take-home, Direction 1: Forecasting against the market / Leakage. Pilot: 2026-10-29; 80-question sample: 2026-11-02. [Code and data guide](README.md) · [Reproduction and audit details](REPRODUCE.md).*

## 1. Question and why it matters

**Do a model's market-relative forecasting scores differ between earlier and later questions, and is better performance associated with retrospective information in its explanations?** Historical forecasting scores can reflect extrapolation, outcome knowledge, intermediate developments, or information embedded in subsequently edited questions. Distinguishing these matters before interpreting a score as evidence of a world model.

In this exploratory sample, Opus performs better relative to the market on earlier questions. That advantage is concentrated among responses flagged for outcome or other post-date information. These observations are consistent with temporal contamination, but do not establish how much of the score difference comes from memorized answers.

## 2. Method

**Data and sampling.** We use 80 resolved binary Polymarket questions, 40 per group. `pre` resolved during 2025-03-03–2026-03-03; `post` resolved after 2026-07-01, with scheduled deadlines more than three days beyond that boundary. Anthropic lists June 2026 as both models' training-data cutoff; this is an exposure proxy, not evidence that any particular outcome entered training. [Model documentation](https://platform.claude.com/docs/en/models/overview).

Filters require at least $20,000 traded volume, a creation-to-resolution horizon of at least 14 days, one market per platform event, topic caps, and a seven-days-before-resolution price in [0.03, 0.97]; keyword rules exclude sports, esports, price bets and parlays. The sample therefore also conditions on later market behavior. It extends a 30-question pilot, retaining 27 questions and adding 53, rather than drawing independently from the full population. A listing cap and subsequently fixed collector bugs left the candidate pool incomplete; sports filters evolved during expansion and one missed golf market was manually removed. The effect on sample composition is unknown. No redraw was performed; this is an exploratory analysis, not independent confirmation. The sampling history is in [REPRODUCE.md](REPRODUCE.md).

**Models and information.** `claude-opus-5-5` and `claude-haiku-5-5` each return one probability and explanation per question, at effort `high`, without tools or web access. The prompt assigns the market's creation **date** as “today,” supplies the question, rules and scheduled deadline, and asks the model to disclose if it believes it knows the outcome. This instruction cannot remove later knowledge. Nor are the supplied rules guaranteed to be their original versions: archived creation-time text is unavailable. Haiku provides a second-model comparison, not an isolated control for memory or capacity.

**Score and timing.** Define each question's paired loss difference as

$$
d_i=(p_{\mathrm{model},i}-y_i)^2-(p_{\mathrm{market},i}-y_i)^2,
\qquad \Delta=\bar d_{\mathrm{post}}-\bar d_{\mathrm{pre}}.
$$

Positive Δ means worse market-relative performance in the later group. The main market baseline is the daily price at or before seven days prior to settlement. **It is later than the model's assigned date:** mean horizons of 60 versus 133 days imply nominal information leads of about 53 versus 126 days. Subtracting market loss does not remove this asymmetry or differences in question composition.

We therefore also report the first available daily price within seven days of creation, and a stricter 24-hour subset. Even these comparisons do not equalize available information: a model may lack legitimate background published after its cutoff but before a question opened. Creation times are also more precise than the date-only instruction.

Intervals use 5,000 within-group bootstrap resamples of the **paired question-level differences**, with percentile endpoints and seed 0. They treat questions as independent, despite possible related topics, and measure neither repeated-generation variability nor sampling bias. The 20,000-draw label-permutation p-values additionally require exchangeability across these observational groups; they are exploratory.

**Text probe.** Haiku judges whether each explanation states or relies on the outcome or events after the assigned date, without seeing the forecast probability. Its legacy field `knows_outcome` is a **later-information flag**, not a validated measure of internal knowledge. It does not inspect the full input or verify claims against dated sources. Familiar older events may be easier to detect; a negative label, especially on newer events, does not establish absence. No blinded human validation has been completed.

## 3. Results: scores and text are separate evidence

| Measure | Earlier (`pre`, n=40) | Later (`post`, n=40) |
|---|---:|---:|
| Market Brier, seven-day baseline | 0.185 | 0.208 |
| Constant 0.5 Brier | 0.250 | 0.250 |
| Opus Brier | 0.109 | 0.309 |
| Opus − market | −0.076 | +0.101 |
| Haiku Brier | 0.227 | 0.344 |
| Haiku − market | +0.042 | +0.136 |
| Opus later-information flags | 30/40 | 0/40 |
| Haiku later-information flags | 10/40 | 0/40 |
| YES rate / mean horizon | 0.575 / 60 days | 0.625 / 133 days |

**Score comparison.** Opus Δ = **+0.177**, 95% bootstrap interval **[+0.055, +0.300]**, permutation p = 0.007. Haiku Δ = +0.094 [−0.057, +0.249], p = 0.243. Opus' later-group loss also exceeds the constant-0.5 baseline on this sample; this alone does not distinguish calibration, discrimination and sampling variation.

**Near-creation baseline.** Prices exist for all 40 earlier and 38 later questions; both missing prices are in `post`. On these same questions, Opus Brier is 0.109/0.302 and market Brier is 0.310/0.230, giving Δ = **+0.272 [+0.156, +0.395]**. Haiku Δ = +0.189 [+0.070, +0.313]. Median price lags are 0.296/0.323 days, maximum lags 2.310/3.973 days, and 10/11 observations lag over a day. Restricting to prices within 24 hours leaves 30/27 questions: Opus Δ = +0.352 [+0.215, +0.499]. On the same 78 questions, the seven-day-baseline estimate is +0.180 [+0.068, +0.300]. Thus the positive association survives these observed time-window checks, but potentially thin opening prices, subset selection and unequal background information remain; the earlier opening-market Brier of 0.310 is itself worse than 0.250.

**Pilot sensitivity.** Opus Δ is +0.096 [−0.124, +0.312] on the 27 retained questions (14 earlier/13 later), and +0.216 [+0.075, +0.370] on the 53 new questions (26/27). The new questions support the same direction but share the evolving selection process. Coarse horizon strata also have positive Opus estimates; this does not eliminate within-stratum confounding. This direction claim does not apply uniformly to Haiku. Full subgroup results are reproducible with `scripts/score.py`.

**Association with text flags.** Using the seven-day market baseline:

| Opus subset | n | Model Brier | Market Brier | Model − market |
|---|---:|---:|---:|---:|
| Earlier, flagged | 30 | 0.073 | 0.192 | −0.118 |
| Earlier, unflagged | 10 | 0.214 | 0.162 | +0.052 |
| Later, unflagged | 40 | 0.309 | 0.208 | +0.101 |

On unflagged questions Δ = **+0.049 [−0.079, +0.175]**. This is a comparison with a different, much smaller earlier subset. It does not estimate the contribution of memory, and an interval spanning zero does not show that the remaining difference is zero. Familiarity, topic, difficulty and detectability may all affect flag membership.

**A source-checked example.** Question 528974 assigns 2025-03-19 as today. Opus nevertheless says the July 2025 FOMC meeting held rates at 4.25–4.50%, with Waller and Bowman dissenting. The [July 30 Federal Reserve statement](https://www.federalreserve.gov/monetarypolicy/monetary20250730a.htm) confirms those details. The saved model input contains the scheduled meeting and resolution rules, but not that outcome or the dissenters. This AI-assisted spot check supports a narrow finding: a date instruction did not prevent a corroborated retrospective outcome assertion in this response. It does not reveal the assertion's internal origin or validate all probe labels. An unflagged response also contains a possible later-event reference; positive and negative audit examples are recorded in [REPRODUCE.md](REPRODUCE.md).

## 4. What would change the interpretation, and the next week

Evidence of actual later-information use would weaken if a blinded, source-backed audit found that apparent retrospective claims were ordinary forecasts based on pre-date information or incorrect recollections. If later facts came from the supplied rules, that would support input contamination and weaken attribution to model-held knowledge. The generality of the score association would weaken if a fresh sample with comparable horizons and information failed to reproduce it. Neither changing the model's cutoff nor moving its assigned date after resolution is, by itself, a decisive test of memory.

With another week I would: (1) redraw from the repaired collector with frozen filters and complete candidate logs; (2) obtain archived input text and dated background packets, comparing model and market at the same timestamp; (3) audit positive and negative labels in both groups, separating final outcomes, intermediate events, input leakage and unverified claims; (4) repeat generations and compare additional models with separately documented cutoffs. A prospective, timestamped forecasting track would provide the clearest check without retrospective date instructions.

The present evidence supports a **group-level score association and concrete retrospective content**, not a causal training-cutoff effect or an estimate of how much advantage memory explains. The groups are separated by several months, so this is not a discontinuity test at the cutoff.

## 5. What I would like to do with you

Build a prospective forecasting evaluation with archived information sets and contemporaneous market baselines, alongside a retrospective diagnostic track. Measure and audit temporal-information violations separately from forecast skill, so improvements in scores can be interpreted with evidence about what information was available.

*AI assistance: Claude Code implemented and ran the experiment and drafted the report under my direction. Codex reviewed the design and code.*
