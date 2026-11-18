"""Pricer par arbre binomial Cox-Ross-Rubinstein (CRR)."""

import numpy as np


def binomial_tree_crr(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Prix d'une option européenne via un arbre CRR à N pas, dividende continu q."""
    kind = option_type.lower()
    if kind not in ("call", "put"):
        raise ValueError("option_type doit être 'call' ou 'put'")

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # Prix possibles du sous-jacent à maturité
    ST = np.array([S0 * (u**j) * (d ** (N - j)) for j in range(N + 1)])

    # Payoff à maturité
    option_values = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)

    # Remontée dans l'arbre
    for _ in range(N):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])

    return float(option_values[0])
