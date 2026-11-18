"""Pricer Monte Carlo (simulation directe de S_T sous la mesure risque-neutre)."""

import numpy as np


def monte_carlo_with_error(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call",
    q: float = 0.0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Retourne (prix, erreur standard) d'une option européenne par Monte Carlo à N tirages."""
    kind = option_type.lower()
    if kind not in ("call", "put"):
        raise ValueError("option_type doit être 'call' ou 'put'")

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0) if kind == "call" else np.maximum(K - S_T, 0)

    discounted = np.exp(-r * T) * payoffs
    stderr = discounted.std(ddof=1) / np.sqrt(N) if N > 1 else float("nan")
    return float(discounted.mean()), float(stderr)


def monte_carlo(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call",
    q: float = 0.0,
    seed: int | None = None,
) -> float:
    """Prix d'une option européenne par Monte Carlo à N tirages."""
    return monte_carlo_with_error(S0, K, T, r, sigma, N, option_type, q, seed)[0]
