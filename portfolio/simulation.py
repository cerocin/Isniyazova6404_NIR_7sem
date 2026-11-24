import numpy as np
import pandas as pd

from portfolio.optimizer import cap_weights


class RandomPortfolios:
    # случайные портфели (Дирихле) для визуальной оценки границы
    def __init__(self, mu, cov, seed=42):
        self.m = mu.to_numpy(float)
        self.c = cov.to_numpy(float)
        self.n = len(self.m)
        self.rng = np.random.default_rng(seed)

    def sample(self, n_samples, max_weight=None):
        cap = 1.0 if max_weight is None else max_weight
        if self.n * cap < 1 - 1e-9:
            raise ValueError("n·w_max < 1: случайные портфели построить нельзя.")
        # разные параметры Дирихле дают и диверсифицированные, и концентрированные портфели
        alphas = self.rng.choice([0.1, 0.3, 1.0, 3.0], size=n_samples)
        w = np.vstack([cap_weights(self.rng.dirichlet(np.full(self.n, a)), cap) for a in alphas])
        risk = np.sqrt(np.einsum("ij,jk,ik->i", w, self.c, w))
        return pd.DataFrame({"risk": risk, "return": w @ self.m, "max_weight": w.max(axis=1)})

    @staticmethod
    def pareto(df):
        # недоминируемые точки: больше доходность при том же или меньшем риске
        d = df.sort_values(["risk", "return"], ascending=[True, False])
        best = d["return"].cummax()
        return d[d["return"] >= best].drop_duplicates(subset="return")
