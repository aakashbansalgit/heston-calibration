"""
Heston model: pricing by Fourier inversion, Monte Carlo for checking, and
calibration to an implied volatility surface.

Everything is written in terms of the forward F and the discount factor, so the
same code works whatever rate and dividend assumptions produced them.

    dS/S = (r - q) dt + sqrt(v) dW1
    dv   = kappa (theta - v) dt + eta sqrt(v) dW2,   d<W1, W2> = rho dt
"""
from dataclasses import astuple, dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

# Gauss-Legendre nodes on [0, U_MAX] for the pricing integral. The Lewis form
# below decays like 1/u^2 even at short maturities, so a fixed grid is enough.
U_MAX = 300.0
_x, _w = np.polynomial.legendre.leggauss(400)
U_NODES = 0.5 * U_MAX * (_x + 1)
U_WEIGHTS = 0.5 * U_MAX * _w


@dataclass
class Params:
    v0: float      # initial variance
    kappa: float   # mean reversion speed
    theta: float   # long-run variance
    eta: float     # volatility of variance
    rho: float     # spot-variance correlation

    def feller(self):
        """2 kappa theta / eta^2. Above 1, variance never touches zero."""
        return 2 * self.kappa * self.theta / self.eta ** 2


def char_func(u, tau, p):
    """Characteristic function of ln(S_T / F).

    Uses the formulation of Albrecher et al. (2007), the "little Heston trap",
    which keeps the complex logarithm on its principal branch. The textbook
    form jumps branches at long maturities, which is the instability the first
    version of this code patched by flipping the sign of d by hand."""
    v0, kappa, theta, eta, rho = astuple(p)
    iu = 1j * u
    b = kappa - rho * eta * iu
    d = np.sqrt(b * b + eta * eta * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * tau)
    C = kappa * theta / eta ** 2 * ((b - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / eta ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def call_price(F, K, tau, df, p):
    """European call by the Lewis (2000) single-integral formula.

    C = df * (F - sqrt(F K) / pi * int_0^inf Re[e^{i u k} phi(u - i/2)] / (u^2 + 1/4) du)
    with k = ln(F / K). Vectorised over K and tau."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    tau = np.broadcast_to(np.asarray(tau, dtype=float), K.shape)
    k = np.log(F / K)[:, None]
    u = U_NODES[None, :]
    phi = char_func(u - 0.5j, tau[:, None], p)
    integrand = np.real(np.exp(1j * u * k) * phi) / (u * u + 0.25)
    integral = integrand @ U_WEIGHTS
    return df * (F - np.sqrt(F * K) / np.pi * integral)


def put_price(F, K, tau, df, p):
    """By put-call parity on the forward."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    return call_price(F, K, tau, df, p) - df * (F - K)


# ---- Black-Scholes on the forward ------------------------------------------

def black(F, K, tau, df, vol, call=True):
    s = vol * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    if call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black_vega(F, K, tau, df, vol):
    s = vol * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return df * F * norm.pdf(d1) * np.sqrt(tau)


def implied_vol(price, F, K, tau, df, call=True):
    """Black implied volatility by Brent's method. NaN if the price is outside
    the no-arbitrage bounds."""
    intrinsic = df * max(F - K, 0.0) if call else df * max(K - F, 0.0)
    upper = df * F if call else df * K
    if not (intrinsic < price < upper):
        return np.nan
    f = lambda v: black(F, K, tau, df, v, call) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10)
    except ValueError:
        return np.nan


# ---- Monte Carlo, for checking the pricer ----------------------------------

def mc_call(F, K, tau, df, p, n_paths=200_000, n_steps=400, seed=0):
    """Full-truncation Euler on ln S and v (Lord, Koekkoek and van Dijk, 2010).
    Returns (price, standard error) for each strike."""
    v0, kappa, theta, eta, rho = astuple(p)
    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    x = np.zeros(n_paths)
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho * rho) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        x += -0.5 * vp * dt + np.sqrt(vp * dt) * z1
        v += kappa * (theta - vp) * dt + eta * np.sqrt(vp * dt) * z2
    ST = F * np.exp(x)
    K = np.atleast_1d(K)
    pay = np.maximum(ST[None, :] - K[:, None], 0.0)
    return df * pay.mean(axis=1), df * pay.std(axis=1) / np.sqrt(n_paths)


# ---- calibration -----------------------------------------------------------

BOUNDS = (
    [1e-4, 0.05, 1e-4, 0.01, -0.999],   # v0, kappa, theta, eta, rho
    [1.0, 15.0, 1.0, 3.0, 0.999],
)


def calibrate(quotes, start=Params(0.02, 2.0, 0.04, 0.8, -0.7)):
    """Fit Heston to market implied vols.

    `quotes` needs columns F, K, tau, df, iv, is_call and weight. Residuals are
    price errors divided by Black vega, which is the implied vol error to first
    order and avoids inverting the model price at every iteration."""
    F, K, tau, df = (quotes[c].to_numpy(float) for c in ("F", "K", "tau", "df"))
    iv = quotes["iv"].to_numpy(float)
    is_call = quotes["is_call"].to_numpy(bool)
    w = np.sqrt(quotes["weight"].to_numpy(float))

    mkt = np.where(is_call, black(F, K, tau, df, iv, True),
                   black(F, K, tau, df, iv, False))
    vega = black_vega(F, K, tau, df, iv)

    # Group by maturity so each call to the pricer shares one forward.
    groups = [np.flatnonzero(tau == t) for t in np.unique(tau)]

    def model_prices(p):
        out = np.empty_like(K)
        for idx in groups:
            c = call_price(F[idx][0], K[idx], tau[idx][0], df[idx][0], p)
            put = c - df[idx] * (F[idx] - K[idx])
            out[idx] = np.where(is_call[idx], c, put)
        return out

    def residuals(x):
        return w * (model_prices(Params(*x)) - mkt) / vega

    fit = least_squares(residuals, astuple(start), bounds=BOUNDS,
                        x_scale="jac", max_nfev=500)
    return Params(*fit.x), fit


def model_iv(quotes, p):
    """Implied vols of the fitted model at each quote, for reporting."""
    out = []
    for row in quotes.itertuples():
        price = (call_price if row.is_call else put_price)(row.F, row.K, row.tau, row.df, p)[0]
        out.append(implied_vol(price, row.F, row.K, row.tau, row.df, row.is_call))
    return np.array(out)
