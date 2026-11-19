<div align="center">

# 🌴 Leonida Rarity Forecaster

### How many GTA VI players earn each trophy in week one — forecast, not guessed.

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aam110/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aam110/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

**Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)** · $250 escrowed · 20 paid places

*One program. Fifty-five trophies. Five quantiles each. Zero network calls.*

</div>

---

## The one idea that matters

An earn rate is a **ratio** — earners over players-tracked — and on launch week
the two sides move for completely different reasons:

| Side | What drives it | Launch-week behaviour |
|---|---|---|
| **Players tracked** | New buyers flooding in | Jumps in big steps every time Exophase refreshes — 638 → 3,868 players in the first refresh alone, diluting *every* rate at once |
| **Earners** | Players actually playing | Creeps up trophy by trophy — fast for Chapter 1, glacial for the Platinum (4 earners at launch, still 4 after the first refresh) |

A forecast that models the rate as one number misses this entirely. The house
baseline just holds the last rate — and scores **~1.51** pinball loss on the
practice cases, with **0% coverage**, because the refresh took every rate
outside its narrow band. This entry models the two sides **separately** and
divides:

```
rate(target) = earners(target) / players(target)
```

---

## How it works

### 1. Collapse the frozen page
Exophase refreshes on its own schedule, so long runs of identical reads are
one frozen page, not fresh evidence. Consecutive identical reads are collapsed
into distinct states, each anchored at the moment it was **first seen**.

### 2. Fit both sides in log time
Each side is a log-linear fit in `x = ln(1 + hours since the first read)`:

- **Players:** `ln(players)` on `x`, slope shrunk toward a prior of 1.0
  (launch morning roughly doubled the tracked population every ~2 hours).
- **Earners, per trophy:** `ln(earners + 0.5)` on `x`, slope shrunk toward a
  class prior (see below). Fitted on the **recent 36-hour window** — the
  launch-night regime is not the day-3 regime — and anchored at the last
  distinct state, so a stale page correctly lengthens the extrapolation step.

### 3. Priors from trophy class
Every trophy is classified from its tier and description keywords. Each class
carries a prior earner-growth slope, a long-run rate cap, and a spread:

| Class | Examples | Growth prior | Cap | Spread |
|---|---|---|---|---|
| Platinum | Vice Got a Price! | 0.05 | 0.35% | 0.55 |
| 100% completion | A Long Road | 0.05 | 0.60% | 0.55 |
| Grind / collectibles | 250 Stunts, 100 locations, 3 card sets | 0.18 | 1.5% | 0.50 |
| Chapter 1 | Paradise Lost | 0.50 | 55% | 0.30 |
| Chapter 2–4 | … | 0.35 / 0.30 / 0.25 | 42 / 32 / 24% | 0.30–0.35 |
| Finale | Complete the final mission | 0.20 | 16% | 0.35 |
| Story missions | Breakfast Run, The Cartel Heist | 0.30 | 18% | 0.35 |
| Stranger strands | Final mission for Andy / Kenton / … | 0.25 | 10% | 0.40 |
| Money | Earn $250,000 laundering… | 0.30 | 6% | 0.40 |
| Misc one-offs | Robberies, exploration, activities | 0.40 | 8% | 0.40 |

The growth priors were calibrated on the **real first refresh** (chapter-1
earners 8 → 19, platinum earners 4 → 4, players 638 → 3,868) and the caps on
**GTA V's long-run rates** (first story trophy ~73%, platinum ~0.1%).

### 4. Guardrails on the median
- **Soft cap:** a fit that overshoots its class asymptote is pulled most of
  the way back.
- **Persistence blend (80/20):** the extrapolated move is blended toward the
  last rate in logit space, because launch-week growth saturates. Skipped
  entirely when the history is flat — then the last rate is a stale page,
  not an anchor, and the modelled move (usually a dilution drop) is the
  whole point.
- **Universal trophies:** a trophy at ≥99% in the last read (the opening
  mission sat at 100% even after the tracked population grew 6×) is pinned
  near 99.7% — the ratio model can't express "every tracked player holds it".

### 5. Honest spreads
Each trophy's logit spread σ is built in quadrature from:

```
σ² = (0.22·√days)² + class² + fit-residual² + (½·players-residual)²
     + 0.45² if history is flat (a refresh is overdue)
     + 0.25² if the last distinct page is >3h stale
```

clamped to [0.40, 1.80]. Quantiles are `median ± z·σ` on the logit scale,
`z = (−1.645, −0.674, 0, 0.674, 1.645)`, rounded to 4 decimals (half-up).

---

## Validation

No server grade exists before submission, so the entry was validated locally
two ways (harness kept outside the repo):

### Practice cases vs the real first refresh
The live Exophase page (19 Nov 11:37 UTC) still showed the 04:30 refresh —
integer-earner analysis puts its denominator at ~3,879 players, vs 3,868 at
the refresh — so its rates serve as a proxy answer for p00/p01:

| Case | This model | House baseline | Baseline (RULES.txt) |
|---|---|---|---|
| p00 | **0.315** (coverage 0.98, miss 0.11pp) | 1.508 (coverage 0.00) | ~1.51 |
| p01 | **0.295** (coverage 0.98, miss 0.10pp) | 1.508 (coverage 0.00) | ~1.51 |

The model anticipates the refresh: chapter 1 forecast 0.54% vs actual 0.49%,
platinum 0.12% vs 0.10% — the baseline held 1.25% and 0.63%.

### Synthetic launch week (final-like: day 2.5 → day 7)
A simulator with power-law player growth, per-class logistic earn-share
curves (half-times from 12h for Chapter 1 to 3,000h for the Platinum),
pre-launch earners from the real launch-night rates, and Exophase refreshes
every ~5h. Ten seeds, truth moves 0.72–0.88 logit:

| | Mean pinball loss | Typical miss | Seeds won |
|---|---|---|---|
| **This model** | **0.389** | 1.31pp | **10 / 10** |
| House baseline | 0.464 | 3.23pp | 0 / 10 |

The simulator's curve family (logistic shares) deliberately differs from the
model's (log-linear fits), so the test is not circular. Blend and spread
parameters were tuned on this backtest: blend 1.0 scored best (0.378), 0.8
was kept as a 20% hedge against a faster-saturating real week (~0.01 cost);
widening the spread beyond 0.22·√days made things worse, so it stayed.

### Format, determinism, runtime
- ✅ 55 rows, one per trophy, quantiles non-decreasing, all in [0, 100]
- ✅ Byte-identical across runs (no randomness, no wall clock)
- ✅ **~0.1 s** for both practice cases (limit: 8 s per case)
- ✅ Output ~2.8 KB (limit: 65,536 bytes)
- ✅ Edge cases pass: single-read history, null rates, 0.00% rates,
  420-read history

---

## Run it locally

```bash
# From the kit directory (input/ holds p00.in and p01.in):
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# → out/p00.out, out/p01.out
```

Requirements: **Python 3.13, standard library only.** No network, no
dependencies, no randomness. The grader runs `python3 /bot/main.py` with
`/input` and `/output`; the `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`
variables are the local stand-ins.

---

## Sources & priors

- Exophase, *Grand Theft Auto VI Trophies — PS5* (live page, read 19 Nov
  2026 11:37 UTC): the first-refresh rates and the still-100% opening trophy.
  <https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/>
- Push Square via GameFAQs (18 Nov 2026): 55 trophies including a Platinum;
  100% completion required, as in every Rockstar title since PS3.
- Polygon / Insider Gaming (15 Nov 2026): ~50-hour campaign, five chapters,
  ~200+ hours for the Platinum (unofficial claims — used only to set the
  platinum/completion half-times far beyond week one).
- PocketPSN, *Grand Theft Auto V Trophies & Stats*: GTA V long-run anchors —
  first story trophy 73.1%, platinum 0.1%, grind trophies 0.1–1%.
- The kit's `RULES.txt` and `input/NOTICE.txt`: refresh behaviour, scoring
  (mean pinball loss on the logit scale), and the house baselines this entry
  is measured against.

---

## Files

| File | What it is |
|---|---|
| `main.py` | The forecasting program — the scored artifact |
| `README.md` | This file |
| `LICENSE` | MIT |

---

<div align="center">

*Built fresh for make.fun · GTA VI launched 19 November 2026 · target 26 November 2026 00:00 UTC*

**Vice City waits for no forecaster. 🌅**

</div>
