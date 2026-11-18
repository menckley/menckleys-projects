# menckleys-projects

Pricers d'options européennes vanilles (call/put) avec dividende continu `q` :

- `pricing/black_scholes.py` : formule fermée Black-Scholes-Merton ;
- `pricing/binomial.py` : arbre binomial Cox-Ross-Rubinstein ;
- `pricing/monte_carlo.py` : simulation Monte Carlo (graine optionnelle, erreur standard disponible).

Le notebook `vanillaoption.ipynb` utilise ces fonctions.

## Installation

```bash
pip install -r requirements.txt -r requirements-dev.txt
# ou, avec uv :
uv sync
```

## Exemple

```python
from pricing import binomial_tree_crr, black_scholes, monte_carlo

black_scholes(100, 100, 1.0, 0.05, 0.2, "call")                   # 10.4506
binomial_tree_crr(100, 100, 1.0, 0.05, 0.2, 1000, "call")         # ≈ 10.45
monte_carlo(100, 100, 1.0, 0.05, 0.2, 100_000, "call", seed=0)    # ≈ 10.4
```

Signature : `(S0, K, T, r, sigma, [N,] option_type, q=0.0)`. Un `option_type` autre que
`"call"`/`"put"` lève `ValueError`.

## Tests et lint

```bash
ruff check .
pytest
```

## Notebook

Lancer Jupyter depuis la racine du dépôt. Pour éviter de versionner les sorties des cellules :

```bash
pip install nbstripout
nbstripout --install
```
