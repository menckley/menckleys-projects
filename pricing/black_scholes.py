"""Pricer Black-Scholes-Merton."""

import numpy as np
from scipy.stats import norm


def black_scholes(
    S0: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0
) -> float:
    """Prix Black-Scholes-Merton d'une option européenne avec dividende continu q."""
    kind = option_type.lower()
    if kind not in ("call", "put"):
        raise ValueError("option_type doit être 'call' ou 'put'")

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if kind == "call":
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return float(price)
