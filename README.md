# GTA VI Trophy Rarity Forecast

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/azizfai/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/azizfai/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of
Exophase-tracked players who will have earned it a week after launch
(2026-11-26), as five quantiles per trophy. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## How it works

`main.py` (Python 3.13, stdlib only) reads each case from the input dir and
writes one forecast file per case. All forecasting happens in logit space:

- Per trophy, a weighted least-squares trend of logit(rate) against
  ln(1 + hours since launch) over the last 24 h of reads, recency-weighted
  with a 12 h half-life. Slopes shrink toward the cross-trophy median slope.
- The median moves at most 2.5 logit units from the last read rate.
- Story-chain trophies (chapters 1–4, final mission, 100% completion,
  platinum) are forced to non-increasing medians.
- Quantiles come from a two-component normal mixture: 75% narrow
  (trend/residual/diffusion uncertainty) and 25% wide (sd 1.20 logit), so the
  5%/95% quantiles stay honest about the chance of a discrete Exophase
  refresh jump while the 25%/75% stay tight.

Runs in ~0.1 s per case, deterministic (two runs byte-identical).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Try make.fun yourself: https://make.fun/r/p5tm223q
