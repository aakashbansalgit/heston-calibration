# heston-calibration

A Heston stochastic volatility pricer, tested against independent methods, and a calibration to a live SPX option chain.

## The pricer

`heston.py` prices European options with the Lewis single-integral formula, which only needs the characteristic function of the log price and decays like 1/u² even at short maturities, so a fixed 400-node Gauss-Legendre grid is enough. The characteristic function uses the "little Heston trap" form of Albrecher et al. (2007), which stays on the principal branch of the complex logarithm. The first version of this code used the textbook form and had to flip the sign of d by hand to stop it jumping branches. Everything is written on the forward and the discount factor, so no rate or dividend assumption is built in.

`tests/test_heston.py` checks it five ways:

- with almost no vol of vol it reproduces Black-Scholes to 1e-4, from 5 weeks to 5 years
- the fixed grid matches scipy's adaptive integration to 1e-6, including a two-week maturity
- a 200,000-path full-truncation Monte Carlo agrees within four standard errors
- call prices fall and are convex in strike, so there is no static arbitrage
- calibrating to a surface priced with known parameters recovers all five to within 2%

## Calibration to SPX

`calibrate_spx.py` takes a snapshot of the SPX chain from Yahoo Finance for seven expiries from three weeks to a year. For each expiry it backs the forward out of put-call parity, regressing call minus put on strike near the money, so the forward comes from the market rather than from an assumed rate and dividend yield. It keeps out-of-the-money quotes between 80% and 120% of the forward, drops strikes with no open interest, quotes wider than 3 vol points and isolated quotes far from their neighbours, and fits all five parameters at once. Residuals are price errors divided by vega, which is the implied vol error to first order, weighted towards tighter quotes.

On the snapshot taken at 2:48 pm New York time on 7 October 2026, with the index at 7,804:

| | |
|---|---|
| Quotes | 1,434 across 7 expiries |
| v0 | 0.0091 (spot vol about 9.5%) |
| kappa | 8.46 |
| theta | 0.0403 (long-run vol about 20%) |
| eta | 1.83 |
| rho | -0.63 |
| Feller ratio | 0.20 |
| RMSE | 0.62 vol points |
| Fit time | about 7 seconds |

| Expiry | Days | RMSE, vol points |
|---|---|---|
| 26 Oct 2026 | 19 | 1.25 |
| 13 Nov 2026 | 37 | 1.16 |
| 18 Dec 2026 | 72 | 0.60 |
| 29 Jan 2027 | 114 | 0.31 |
| 31 Mar 2027 | 175 | 0.28 |
| 30 Jun 2027 | 266 | 0.41 |
| 30 Sep 2027 | 358 | 0.47 |

`figures/smiles_2026-10-07.png` shows the fitted smiles against the market.

Beyond three months the fit is within half a vol point. The short end is where Heston breaks down, and it breaks in the usual way. A 19-day smile this steep needs more short-dated skew than one diffusion can produce, so the fit pushes kappa to 8.5 and eta to 1.8 to bend the short end, and still undershoots the far downside puts by about 5 vol points. The Feller ratio of 0.2 says the fitted variance process hits zero, which is common for equity index fits and is the cost of matching that skew.

I also score the fit the way I would score any model against market quotes: the share of strikes where the model's implied vol lands inside the bid-ask. SPX is liquid enough that the median quote is only 0.05 vol points wide, so with 0.6 points of error Heston lands inside on about 4% of strikes. RMSE says the fit is reasonable; the bid-ask test says a trader could not quote off it. Rough volatility or a jump component is the usual next step.

## Running it

```
pip install numpy scipy pandas matplotlib yfinance
python -m pytest tests          # or: python tests/test_heston.py
python calibrate_spx.py         # refit the saved snapshot
python calibrate_spx.py --download   # take a new snapshot first
```
