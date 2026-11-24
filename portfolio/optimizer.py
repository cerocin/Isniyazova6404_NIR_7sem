import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_NAMES = [
    "1. Равные веса",
    "2. Макс. доход. (риск ≤ R)",
    "3. Мин. риск (доход. ≥ r)",
    "4. Макс. доход. (риск ≤ R, доля ≤ w)",
    "5. Мин. риск (доход. ≥ r, доля ≤ w)",
    "6. Макс. Шарп",
]


class InfeasibleError(ValueError):
   
    pass


def cap_weights(w, cap):
    
    w = np.asarray(w, dtype=float).copy()
    if cap >= 1.0:
        return w
    over = w > cap
    excess = float((w[over] - cap).sum())
    if excess <= 0:
        return w
    w[over] = cap
    room = np.where(over, 0.0, cap - w)
    if room.sum() < excess - 1e-12:
        raise InfeasibleError("n·w_max < 1: веса не могут в сумме дать 1.")
    return w + excess * room / room.sum()


class PortfolioOptimizer:
    def __init__(self, mu, cov, rf=0.0, tol=1e-6, max_iter=500, seed=42):
        if not mu.index.equals(cov.index) or not cov.index.equals(cov.columns):
            raise ValueError("Индексы mu и cov не совпадают.")
        if mu.isna().any() or cov.isna().any().any():
            raise ValueError("В mu или cov есть NaN.")
        self.names = list(mu.index)
        self.n = len(self.names)
        self.m = mu.to_numpy(float)
        self.c = cov.to_numpy(float)
        self.rf, self.tol, self.max_iter = rf, tol, max_iter
        self.rng = np.random.default_rng(seed)

    
    def ret(self, w):
        return float(np.dot(w, self.m))

    def risk(self, w):
        w = np.asarray(w, float)
        return float(np.sqrt(max(w @ self.c @ w, 0.0)))

    def sharpe(self, w):
        s = self.risk(w)
        if s < 1e-12:
            raise ZeroDivisionError("Риск портфеля равен нулю: коэффициент Шарпа не определён.")
        return (self.ret(w) - self.rf) / s

   
    def _check_cap(self, cap):
        if cap is not None and self.n * cap < 1 - 1e-9:
            raise InfeasibleError(f"n·w_max = {self.n}·{cap} < 1: доли активов не могут давать в сумме 1.")

    def max_return_weights(self, cap=None):
        
        cap = 1.0 if cap is None else cap
        self._check_cap(cap)
        w, left = np.zeros(self.n), 1.0
        for i in np.argsort(-self.m):
            take = min(cap, left)
            w[i], left = take, left - take
            if left <= 1e-12:
                break
        return w

    def _is_ok(self, w, cons, cap):
        if w.max() > cap + self.tol or w.min() < -self.tol:
            return False
        for c in cons:
            v = c["fun"](w)
            if (c["type"] == "eq" and abs(v) > self.tol) or (c["type"] == "ineq" and v < -self.tol):
                return False
        return True

    def _solve(self, fun, jac, cons, cap, what):
        cap = 1.0 if cap is None else cap
        starts = [np.full(self.n, 1.0 / self.n)]
        starts += [cap_weights(self.rng.dirichlet(np.ones(self.n)), cap) for _ in range(3)]
        msg = ""
        for x0 in starts:
            res = minimize(fun, x0, jac=jac, method="SLSQP", bounds=[(0.0, cap)] * self.n,
                           constraints=cons, options={"maxiter": self.max_iter, "ftol": 1e-12})
            msg = res.message
            if not res.success:
                continue
            w = np.clip(res.x, 0.0, cap)
            w = w / w.sum()
            if self._is_ok(w, cons, cap):
                return w
        raise InfeasibleError(f"{what}: решение не найдено ({msg})")

    @staticmethod
    def _budget():
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}

    
    def equal_weights(self):
        return np.full(self.n, 1.0 / self.n)

    def min_risk(self, target_return=None, max_weight=None):
        self._check_cap(max_weight)
        cons = [self._budget()]
        if target_return is not None:
            bound = self.ret(self.max_return_weights(max_weight))
            if target_return > bound + 1e-9:
                raise InfeasibleError(f"целевая доходность {target_return:.2%} выше максимально "
                                      f"достижимой {bound:.2%}")
            cons.append({"type": "ineq", "fun": lambda w: self.ret(w) - target_return,
                         "jac": lambda w: self.m})
        return self._solve(lambda w: w @ self.c @ w, lambda w: 2 * self.c @ w,
                           cons, max_weight, "мин. риск")

    def max_return(self, max_risk, max_weight=None):
        gmv_risk = self.risk(self.min_risk(None, max_weight))
        if max_risk < gmv_risk - 1e-9:
            raise InfeasibleError(f"риск {max_risk:.2%} ниже минимально достижимого {gmv_risk:.2%}")
        cons = [self._budget(),
                {"type": "ineq", "fun": lambda w: max_risk ** 2 - w @ self.c @ w,
                 "jac": lambda w: -2 * self.c @ w}]
        return self._solve(lambda w: -self.m @ w, lambda w: -self.m, cons, max_weight, "макс. доходность")

    def max_sharpe(self, max_weight=None):
        bound = self.ret(self.max_return_weights(max_weight))
        if bound <= self.rf:
            raise InfeasibleError(f"ни один портфель не даёт доходность выше rf={self.rf:.2%} "
                                  f"(максимум {bound:.2%})")

        def neg_sharpe(w):
            s = np.sqrt(max(w @ self.c @ w, 1e-18))
            return -(self.m @ w - self.rf) / s

        def grad(w):
            var = max(w @ self.c @ w, 1e-18)
            s = np.sqrt(var)
            return -(self.m / s - (self.m @ w - self.rf) * (self.c @ w) / s ** 3)

        return self._solve(neg_sharpe, grad, [self._budget()], max_weight, "макс. Шарп")

    def solve_models(self, max_risk, target_return, max_weight):
        
        jobs = [
            lambda: self.equal_weights(),
            lambda: self.max_return(max_risk),
            lambda: self.min_risk(target_return),
            lambda: self.max_return(max_risk, max_weight),
            lambda: self.min_risk(target_return, max_weight),
            lambda: self.max_sharpe(),
        ]
        weights, status = {}, {}
        for name, job in zip(MODEL_NAMES, jobs):
            try:
                weights[name] = pd.Series(job(), index=self.names)
                status[name] = "OK"
            except InfeasibleError as e:
                status[name] = f"нет решения: {e}"
        return weights, status

    def efficient_frontier(self, n_points=30, max_weight=None):
        
        w_gmv = self.min_risk(None, max_weight)
        w_top = self.max_return_weights(max_weight)
        lo, hi = self.ret(w_gmv), self.ret(w_top)
        rows = [(self.risk(w_gmv), lo)]
        for t in np.linspace(lo, hi, n_points)[1:-1]:
            try:
                w = self.min_risk(t, max_weight)
            except InfeasibleError:
                continue
            rows.append((self.risk(w), self.ret(w)))
        rows.append((self.risk(w_top), hi))
        df = pd.DataFrame(rows, columns=["risk", "return"]).drop_duplicates()
        return df.sort_values("risk").reset_index(drop=True)
