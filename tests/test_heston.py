"""Checks on the pricer and the calibration.

Run from the repo root: python -m pytest tests  (or python tests/test_heston.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heston as h

P = h.Params(v0=0.04, kappa=1.5, theta=0.04, eta=0.6, rho=-0.7)


def test_black_limit():
    """With almost no vol of vol and v0 = theta, Heston is Black at sqrt(v0)."""
    p = h.Params(v0=0.04, kappa=1.0, theta=0.04, eta=1e-3, rho=0.0)
    K = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
    for tau in (0.1, 1.0, 5.0):
        heston = h.call_price(100.0, K, tau, 0.97, p)
        black = h.black(100.0, K, tau, 0.97, 0.2)
        assert np.allclose(heston, black, atol=1e-4), (tau, heston - black)


def test_quadrature_matches_adaptive_integration():
    """The fixed Gauss-Legendre grid agrees with scipy's adaptive quad,
    including a two-week and a five-year maturity."""
    F, df = 100.0, 1.0
    for tau in (0.04, 1.0, 5.0):
        for K in (80.0, 100.0, 125.0):
            k = np.log(F / K)
            f = lambda u: np.real(np.exp(1j * u * k) * h.char_func(u - 0.5j, tau, P)) / (u * u + 0.25)
            ref = df * (F - np.sqrt(F * K) / np.pi * quad(f, 0, np.inf, limit=500)[0])
            assert abs(h.call_price(F, K, tau, df, P)[0] - ref) < 1e-6, (tau, K)


def test_monte_carlo_agrees():
    K = np.array([80.0, 100.0, 120.0])
    exact = h.call_price(100.0, K, 1.0, 1.0, P)
    mc, se = h.mc_call(100.0, K, 1.0, 1.0, P)
    assert np.all(np.abs(exact - mc) < 4 * se + 0.02), (exact, mc, se)


def test_no_static_arbitrage_across_strikes():
    """Call prices fall and are convex in strike."""
    K = np.linspace(50, 200, 151)
    c = h.call_price(100.0, K, 0.5, 0.98, P)
    assert np.all(np.diff(c) < 0)
    assert np.all(np.diff(c, 2) > -1e-10)


def test_calibration_recovers_known_parameters():
    """Price a surface with known parameters, then calibrate to it from a
    different starting point."""
    true = h.Params(v0=0.025, kappa=2.5, theta=0.05, eta=0.9, rho=-0.75)
    rows = []
    for tau in (0.08, 0.25, 0.5, 1.0, 2.0):
        F, df = 100.0 * np.exp(0.02 * tau), np.exp(-0.04 * tau)
        for K in F * np.linspace(0.75, 1.25, 11):
            call = K >= F
            price = (h.call_price if call else h.put_price)(F, K, tau, df, true)[0]
            rows.append(dict(F=F, K=K, tau=tau, df=df, is_call=call, weight=1.0,
                             iv=h.implied_vol(price, F, K, tau, df, call)))
    # Far out-of-the-money short-dated calls are worth less than the pricer's
    # rounding error, so they have no implied vol and drop out, as they would
    # in a real chain.
    quotes = pd.DataFrame(rows).dropna()
    fit, _ = h.calibrate(quotes)
    got, want = np.array(h.astuple(fit)), np.array(h.astuple(true))
    assert np.allclose(got, want, rtol=0.02, atol=0.002), (got, want)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
