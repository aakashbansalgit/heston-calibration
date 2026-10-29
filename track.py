"""
Weekly recalibration to a fresh SPX snapshot, kept as a time series.

    python track.py

Takes a new snapshot (during US market hours, so the quotes are live), fits
Heston, appends one row to results/history.csv and redraws
figures/parameter_history.png and figures/smiles_latest.png. A snapshot that
looks broken (too few clean quotes, or an RMSE far outside the usual range)
is reported and not recorded.

The question it answers over time is how stable the fitted parameters are.
Heston is known to trade kappa off against eta when it bends the short end,
so a fit that matches the surface well each week can still move its
parameters a lot from one week to the next.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import calibrate_spx as c
import heston as h

ROOT = Path(__file__).parent
HISTORY = ROOT / "results" / "history.csv"
MIN_QUOTES = 400
MAX_RMSE = 3.0      # vol points
README = ROOT / "README.md"
START, END = "<!-- history:start -->", "<!-- history:end -->"


def market_open_today():
    import yfinance as yf
    last = yf.Ticker("^SPX").history(period="5d").index[-1].date()
    return last == datetime.now(c.NY).date()


def fit_snapshot():
    if not market_open_today():
        sys.exit("no SPX session today, not recording")
    path = c.download_snapshot()
    q = c.prepare(pd.read_csv(path))
    if len(q) < MIN_QUOTES:
        sys.exit(f"only {len(q)} clean quotes, not recording")
    t0 = datetime.now()
    p, _ = h.calibrate(q)
    q, by_exp, summary = c.report(q, p, (datetime.now() - t0).total_seconds())
    if summary["rmse_vol_pts"] > MAX_RMSE:
        sys.exit(f"RMSE {summary['rmse_vol_pts']} vol points, not recording")
    return q, by_exp, summary


def append(summary):
    row = pd.DataFrame([summary]).drop(columns="fit_seconds")
    row.insert(0, "date", summary["snapshot"][:10])
    hist = pd.concat([pd.read_csv(HISTORY), row]) if HISTORY.exists() else row
    hist = hist.drop_duplicates("date", keep="last").sort_values("date")
    hist.to_csv(HISTORY, index=False)
    return hist


def plot_history(hist):
    d = pd.to_datetime(hist["date"])
    panels = [("v0", "v0"), ("theta", "theta"), ("kappa", "kappa"),
              ("eta", "eta (vol of vol)"), ("rho", "rho"), ("rmse_vol_pts", "RMSE, vol points")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(d, hist[col], marker="o", ms=3, lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "parameter_history.png", dpi=110)


def update_readme(hist):
    recent = hist.tail(8).iloc[::-1]
    lines = ["| Date | SPX | v0 | kappa | theta | eta | rho | RMSE | Inside bid-ask |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in recent.iterrows():
        lines.append(f"| {r.date} | {r.spot:,.0f} | {r.v0:.4f} | {r.kappa:.2f} | {r.theta:.4f} "
                     f"| {r.eta:.2f} | {r.rho:.2f} | {r.rmse_vol_pts:.2f} | {r.inside_bid_ask:.0%} |")
    n = len(hist)
    lines.append("")
    lines.append(f"{n} weekly fit{'s' if n > 1 else ''} since {hist.date.iloc[0]}. "
                 "The full series is in `results/history.csv`.")
    text = README.read_text(encoding="utf-8")
    before, rest = text.split(START)
    after = rest.split(END)[1]
    README.write_text(before + START + "\n" + "\n".join(lines) + "\n" + END + after,
                      encoding="utf-8")


def main():
    q, by_exp, summary = fit_snapshot()
    hist = append(summary)
    by_exp.to_csv(ROOT / "results" / "latest_by_expiry.csv", float_format="%.4f")
    c.plot(q, ROOT / "figures" / "smiles_latest.png")
    plot_history(hist)
    update_readme(hist)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
