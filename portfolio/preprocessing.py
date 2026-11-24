import numpy as np
import pandas as pd


class ReturnsBuilder:
   
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _label(ratio):
        k = ratio if ratio >= 1 else 1.0 / ratio
        n = round(k)
        if n >= 2 and abs(k - n) / n <= 0.10:
            return f"сплит/консолидация ≈{n}:1"
        return "аномальный скачок"

    def build(self, prices):
        prices = prices.sort_index()
        
        rets = prices / prices.shift(1) - 1.0

        days = prices.index.to_series().diff().dt.days.to_numpy()
        gap_rows = np.nan_to_num(days, nan=0) > self.cfg.max_gap_days
        n_gap = int(gap_rows.sum())
        if n_gap:
            print(f"  Разрывов календаря > {self.cfg.max_gap_days} дн.: {n_gap} (доходности за них исключены)")
            rets.loc[gap_rows] = np.nan

        
        jump = np.abs(np.log1p(rets)) > self.cfg.jump_threshold
        rows, cols = np.where(jump.to_numpy())
        log = []
        for i, j in zip(rows, cols):
            ratio = 1.0 + rets.iat[i, j]
            log.append({"Дата": rets.index[i].date(), "Тикер": rets.columns[j],
                        "Отношение цен": round(ratio, 4), "Тип события": self._label(ratio)})
        jump_log = pd.DataFrame(log, columns=["Дата", "Тикер", "Отношение цен", "Тип события"])
        rets = rets.mask(jump)
        print(f"[ДОХОДНОСТИ] {rets.shape[0]} дат x {rets.shape[1]} активов; "
              f"исключено скачков цены: {len(jump_log)}")
        return rets.dropna(how="all"), jump_log

    def select_assets(self, rets, verbose=True):
       
        cnt = rets.count()
        coverage = cnt / cnt.max()
        ok = (coverage >= self.cfg.min_coverage) & (cnt >= self.cfg.min_obs) & (rets.std() > 0)
        dropped = list(rets.columns[~ok.to_numpy()])
        selected = list(rets.columns[ok.to_numpy()])
        if verbose:
            print(f"[ОТБОР АКТИВОВ] оставлено {len(selected)}, исключено {len(dropped)} "
                  f"(покрытие < {self.cfg.min_coverage:.0%} или < {self.cfg.min_obs} наблюдений): {dropped}")
        if len(selected) < self.cfg.min_assets:
            raise ValueError(f"После отбора осталось {len(selected)} активов, нужно минимум {self.cfg.min_assets}.")
        return selected


class MomentsEstimator:
   
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def nearest_psd(cov, eps=1e-8):
       
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
        if vals.min() >= eps:
            return cov, False
        vals = np.clip(vals, eps, None)
        fixed = vecs @ np.diag(vals) @ vecs.T
        d = np.sqrt(np.diag(fixed))
        fixed = fixed / np.outer(d, d)
        return fixed * np.outer(sd, sd), True

    @staticmethod
    def shrink_mean(mu, cov, t_obs):
        
        n = len(mu)
        ones = np.ones(n)
        inv_ones = np.linalg.solve(cov, ones)
        mu0 = float(ones @ np.linalg.solve(cov, mu) / (ones @ inv_ones))
        d = mu - mu0
        q = float(t_obs * d @ np.linalg.solve(cov, d))
        lam = (n + 2) / ((n + 2) + q)
        return (1 - lam) * mu + lam * mu0, lam

    def estimate(self, rets, shrink=None):
        shrink = self.cfg.shrink_mu if shrink is None else shrink
        mu_d = rets.mean().to_numpy()
        cov_d = rets.cov(min_periods=self.cfg.min_pair_obs)
        if cov_d.isna().any().any():
            raise ValueError("Для некоторых пар активов слишком мало общих наблюдений "
                             f"(< {self.cfg.min_pair_obs}) — ковариация не определена.")
        names = list(rets.columns)
        cov_v, repaired = self.nearest_psd(cov_d.to_numpy())
        lam = 0.0
        if shrink:
            mu_d, lam = self.shrink_mean(mu_d, cov_v, int(rets.count().median()))
        k = self.cfg.trading_days
        mu = pd.Series(mu_d * k, index=names)
        cov = pd.DataFrame(cov_v * k, index=names, columns=names)
        info = {"psd_repaired": repaired, "shrink_lambda": lam}
        print(f"[ОЦЕНКА ПАРАМЕТРОВ] активов: {len(names)}, коррекция PSD: {'да' if repaired else 'нет'}, "
              f"интенсивность сжатия mu: {lam:.3f}")
        return mu, cov, info
