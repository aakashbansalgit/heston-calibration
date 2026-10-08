"""
Calibrate Heston to one snapshot of the SPX option chain.

    python calibrate_spx.py              # uses the newest file in data/
    python calibrate_spx.py --download   # takes a fresh snapshot from Yahoo first

Forwards and discount factors are backed out of put-call parity for each
expiry, so no rate or dividend assumption goes in. The fit uses out-of-the-
money quotes only and is scored two ways: the RMSE of implied vol in vol
points, and the share of strikes where the model's implied vol lands inside
the market's bid-ask.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import heston as h

ROOT = Path(__file__).parent
DATA = ROOT / "data"
NY = ZoneInfo("America/New_York")
TARGET_YEARS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
MONEYNESS = (0.80, 1.20)


def download_snapshot():
    import yfinance as yf
    tk = yf.Ticker("^SPX")
    now = datetime.now(NY)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])

    expiries = pd.to_datetime(pd.Series(tk.options))
    years = (expiries - pd.Timestamp(now.date())).dt.days / 365
    chosen = sorted({expiries[(years - t).abs().idxmin()] for t in TARGET_YEARS})

    rows = []
    for exp in chosen:
        chain = tk.option_chain(exp.strftime("%Y-%m-%d"))
        for kind, df in (("C", chain.calls), ("P", chain.puts)):
            df = df[["strike", "bid", "ask", "volume", "openInterest"]].copy()
            df["type"], df["expiry"] = kind, exp.date()
            rows.append(df)
    raw = pd.concat(rows, ignore_index=True)
    raw["spot"], raw["snapshot"] = spot, now.isoformat(timespec="minutes")
    DATA.mkdir(exist_ok=True)
    path = DATA / f"spx_chain_{now:%Y-%m-%d}.csv"
    raw.to_csv(path, index=False)
    return path


def forward_from_parity(g):
    """Regress C - P on K near the money: C - P = df * F - df * K."""
    calls = g[g.type == "C"].set_index("strike")
    puts = g[g.type == "P"].set_index("strike")
    both = calls.join(puts, lsuffix="_c", rsuffix="_p", how="inner")
    both = both[(both.bid_c > 0) & (both.bid_p > 0)]
    spot = g.spot.iloc[0]
    near = both[(both.index / spot).to_series().between(0.95, 1.05).values]
    diff = (near.bid_c + near.ask_c) / 2 - (near.bid_p + near.ask_p) / 2
    slope, intercept = np.polyfit(near.index.to_numpy(float), diff.to_numpy(float), 1)
    df = -slope
    return intercept / df, df, len(near)


def clean(raw):
    """Drop one-sided and crossed quotes, and strikes nobody holds.

    Strikes with zero open interest are often quoted by a market maker's model
    rather than by trading, and in this data some sit several points above
    their neighbours, which is not arbitrage-free. If a strike appears twice,
    keep the tighter quote."""
    raw = raw[(raw.bid > 0) & (raw.ask > raw.bid) & (raw.openInterest > 0)].copy()
    raw["spread"] = raw.ask - raw.bid
    raw = raw.sort_values("spread").drop_duplicates(["expiry", "type", "strike"])
    return raw.drop(columns="spread")


def prepare(raw):
    raw = clean(raw)
    snap = datetime.fromisoformat(raw.snapshot.iloc[0])
    out = []
    for exp, g in raw.groupby("expiry"):
        close = datetime.combine(pd.Timestamp(exp).date(), datetime.min.time(),
                                 NY).replace(hour=16)
        tau = (close - snap).total_seconds() / (365 * 86400)
        if tau < 7 / 365:
            continue
        F, df, n = forward_from_parity(g)
        g = g.assign(F=F, df=df, tau=tau, K=g.strike.astype(float))
        otm = ((g.type == "P") & (g.K < F)) | ((g.type == "C") & (g.K >= F))
        g = g[otm & (g.bid > 0) & (g.ask > g.bid)]
        g = g[(g.K / F).between(*MONEYNESS)]
        g = g.assign(is_call=g.type == "C", mid=(g.bid + g.ask) / 2)
        for col, px in (("iv", "mid"), ("iv_bid", "bid"), ("iv_ask", "ask")):
            g[col] = [h.implied_vol(r[px], r.F, r.K, r.tau, r.df, r.is_call)
                      for _, r in g.iterrows()]
        out.append(g)
    q = pd.concat(out).dropna(subset=["iv", "iv_bid", "iv_ask"])
    # A bid-ask wider than 3 vol points is a stale or indicative quote.
    q = q[q.iv_ask - q.iv_bid < 0.03]
    # Drop isolated quotes more than 1.5 vol points from the median of their
    # seven nearest strikes. A real smile is smooth at this spacing.
    q = q.sort_values(["expiry", "K"])
    local = q.groupby("expiry")["iv"].transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median())
    q = q[(q.iv - local).abs() < 0.015]
    # Wide quotes carry less information about where the market really is.
    q["weight"] = 1 / np.maximum(q.iv_ask - q.iv_bid, 0.005)
    return q.reset_index(drop=True)


def report(q, p, fit_time):
    q = q.assign(model_iv=h.model_iv(q, p))
    err = (q.model_iv - q.iv) * 100
    inside = (q.model_iv >= q.iv_bid) & (q.model_iv <= q.iv_ask)
    by_exp = q.assign(err=err, inside=inside).groupby("expiry").agg(
        tau=("tau", "first"), forward=("F", "first"), quotes=("K", "size"),
        rmse_vol_pts=("err", lambda e: np.sqrt(np.mean(e ** 2))),
        inside_bid_ask=("inside", "mean"))
    summary = {
        "snapshot": q.snapshot.iloc[0], "spot": q.spot.iloc[0],
        "quotes": len(q), "expiries": q.expiry.nunique(),
        **{k: round(v, 4) for k, v in vars(p).items()},
        "feller_ratio": round(p.feller(), 3),
        "rmse_vol_pts": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "inside_bid_ask": round(float(inside.mean()), 3),
        "fit_seconds": round(fit_time, 1),
    }
    return q, by_exp, summary


def plot(q, path):
    exps = sorted(q.expiry.unique())
    cols = 4
    rows = int(np.ceil(len(exps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, exp in zip(axes.flat, exps):
        g = q[q.expiry == exp].sort_values("K")
        m = g.K / g.F
        ax.fill_between(m, g.iv_bid * 100, g.iv_ask * 100, color="0.85", label="bid-ask")
        ax.plot(m, g.iv * 100, ".", ms=3, color="0.3", label="mid")
        ax.plot(m, g.model_iv * 100, color="C3", lw=1.2, label="Heston")
        ax.set_title(f"{exp}  ({g.tau.iloc[0] * 365:.0f} days)", fontsize=9)
        ax.set_xlabel("K / F", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(exps):]:
        ax.axis("off")
    axes.flat[0].set_ylabel("implied vol, %")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    path = download_snapshot() if args.download else max(DATA.glob("spx_chain_*.csv"))
    raw = pd.read_csv(path)
    q = prepare(raw)

    t0 = datetime.now()
    p, fit = h.calibrate(q)
    fit_time = (datetime.now() - t0).total_seconds()
    q, by_exp, summary = report(q, p, fit_time)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stem = path.stem.replace("spx_chain_", "")
    (out / f"fit_{stem}.json").write_text(json.dumps(summary, indent=2))
    by_exp.to_csv(out / f"fit_by_expiry_{stem}.csv", float_format="%.4f")
    (ROOT / "figures").mkdir(exist_ok=True)
    plot(q, ROOT / "figures" / f"smiles_{stem}.png")

    print(json.dumps(summary, indent=2))
    print(by_exp.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
