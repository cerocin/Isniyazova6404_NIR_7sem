import dataclasses

import numpy as np
import pandas as pd

from portfolio.data_loader import CsvStore, MoexLoader
from portfolio.optimizer import MODEL_NAMES, InfeasibleError, PortfolioOptimizer
from portfolio.plotting import Plotter
from portfolio.preprocessing import MomentsEstimator, ReturnsBuilder
from portfolio.report import ReportWriter
from portfolio.simulation import RandomPortfolios
from portfolio.utils import StageTimer, package_versions
from portfolio.validator import PriceValidator

SUMMARY_COLS = ["Доходность", "Риск", "Шарп", "Активов", "Статус"]


class PortfolioPipeline:
 
    def __init__(self, cfg):
        self.cfg = cfg
        self.builder = ReturnsBuilder(cfg)
        self.estimator = MomentsEstimator(cfg)

  
    def _load(self):
        cfg = self.cfg
        if cfg.input_csv:
            return CsvStore(cfg.input_csv).read()
        cache = cfg.data_dir / "raw_prices.csv"
        try:
            prices = MoexLoader(cfg).load_all()
            CsvStore(cache).write(prices)
            return prices
        except ConnectionError as e:
            print(f"[!] {e} Пробую локальную копию: {cache}")
            return CsvStore(cache).read()

    def _slice(self, prices):
      
        prices = prices.loc[self.cfg.start:self.cfg.end].dropna(axis=1, how="all")
        if prices.empty:
            raise ValueError("В выбранном временном срезе нет данных.")
        span = (prices.index[-1] - prices.index[0]).days
        if span < self.cfg.min_span_days:
            raise ValueError(f"Период данных {span} дн. короче минимума {self.cfg.min_span_days} дн.")
        return prices

  
    def _summary(self, opt, weights, status):
        rows = {}
        for name in MODEL_NAMES:
            if name not in weights:
                rows[name] = [np.nan, np.nan, np.nan, 0, status[name]]
                continue
            w = weights[name].to_numpy()
            try:
                sharpe = opt.sharpe(w)
            except ZeroDivisionError:
                sharpe = np.nan
            rows[name] = [opt.ret(w), opt.risk(w), sharpe, int((w > 0.001).sum()), status[name]]
        return pd.DataFrame.from_dict(rows, orient="index", columns=SUMMARY_COLS)

    def _holdout(self, rets):
       
        cfg = self.cfg
        if cfg.holdout <= 0:
            return None
        cut = int(len(rets) * (1 - cfg.holdout))
        train, test = rets.iloc[:cut], rets.iloc[cut:]
        try:
            sel = self.builder.select_assets(train, verbose=False)
            sel = [a for a in sel if test[a].count() >= cfg.min_pair_obs]
            if len(sel) < cfg.min_assets:
                raise ValueError("мало активов с историей в тестовой части")
            mu_tr, cov_tr, _ = self.estimator.estimate(train[sel])
            mu_te, cov_te, _ = self.estimator.estimate(test[sel], shrink=False)
        except ValueError as e:
            print(f"[!] Проверка на отложенной выборке пропущена: {e}")
            return None
        opt_tr = PortfolioOptimizer(mu_tr, cov_tr, cfg.rf, seed=cfg.seed)
        opt_te = PortfolioOptimizer(mu_te, cov_te, cfg.rf, seed=cfg.seed)
        weights, status = opt_tr.solve_models(cfg.max_risk, cfg.target_return, cfg.max_weight)
        rows = {}
        for name in MODEL_NAMES:
            if name not in weights:
                rows[name] = [np.nan] * 5
                continue
            w = weights[name].to_numpy()
            try:
                s_tr = opt_tr.sharpe(w)
                s_te = opt_te.sharpe(w)
            except ZeroDivisionError:
                s_tr = s_te = np.nan
            rows[name] = [opt_tr.ret(w), opt_te.ret(w), opt_tr.risk(w), opt_te.risk(w), s_te]
            rows[name].insert(4, s_tr)
        cols = ["Доходность (обучение)", "Доходность (тест)", "Риск (обучение)", "Риск (тест)",
                "Шарп (обучение)", "Шарп (тест)"]
        print(f"[ОТЛОЖЕННАЯ ВЫБОРКА] обучение: {len(train)} дней, тест: {len(test)} дней")
        return pd.DataFrame.from_dict(rows, orient="index", columns=cols)

   
    def run(self):
        cfg, timer = self.cfg, StageTimer()
        with timer.stage("Загрузка данных"):
            print("=== Загрузка данных ===")
            prices = self._load()
        with timer.stage("Проверка и очистка таблицы"):
            print("=== Проверка таблицы цен ===")
            prices, vrep = PriceValidator(cfg).validate(prices)
            prices = self._slice(prices)
        with timer.stage("Доходности и отбор активов"):
            print("=== Доходности ===")
            rets, jumps = self.builder.build(prices)
            if len(rets) < 100:
                raise ValueError(f"Слишком мало наблюдений: {len(rets)} (нужно минимум 100).")
            selected = self.builder.select_assets(rets)
            rets_sel = rets[selected]
        with timer.stage("Оценка mu и Sigma"):
            mu, cov, info = self.estimator.estimate(rets_sel)
        with timer.stage("Оптимизация (6 моделей)"):
            print("=== Оптимизация ===")
            opt = PortfolioOptimizer(mu, cov, cfg.rf, seed=cfg.seed)
            weights, status = opt.solve_models(cfg.max_risk, cfg.target_return, cfg.max_weight)
            summary = self._summary(opt, weights, status)
        with timer.stage("Граница и Монте-Карло"):
            print("=== Граница и случайные портфели ===")
            frontier = opt.efficient_frontier(cfg.frontier_points, None)
            try:
                frontier_cap = opt.efficient_frontier(cfg.frontier_points, cfg.max_weight)
            except InfeasibleError as e:
                print(f"[!] Граница с ограничением доли недоступна: {e}")
                frontier_cap = None
            mc = RandomPortfolios(mu, cov, cfg.seed).sample(cfg.mc_samples)
            pareto = RandomPortfolios.pareto(mc)
        with timer.stage("Отложенная выборка"):
            oos = self._holdout(rets)

        print("\n--- РЕЗУЛЬТАТЫ ---")
        print(summary.round(4).to_string())
        if oos is not None:
            print("\n--- ОТЛОЖЕННАЯ ВЫБОРКА ---")
            print(oos.round(3).to_string())

        with timer.stage("Сохранение результатов"):
            print("\n=== Сохранение ===")
            writer = ReportWriter(cfg.results_dir)
            writer.write_weights(weights)
            writer.write_summary(summary)
            tables = {"Итоги": (summary, True),
                      "Веса": (pd.DataFrame(weights), True),
                      "Граница": (frontier, False),
                      "Монте-Карло": (mc.drop(columns="max_weight"), False),
                      "Парето МК": (pareto.drop(columns="max_weight"), False),
                      "Скачки цен": (jumps, False),
                      "Проверка данных": (vrep.to_frame(), False)}
            if frontier_cap is not None:
                tables["Граница с огр. доли"] = (frontier_cap, False)
            if oos is not None:
                tables["Вне выборки"] = (oos, True)
            writer.write_excel(tables)
        with timer.stage("Построение графиков"):
            print("\n=== Графики ===")
            self._plots(frontier, frontier_cap, mc, summary, weights, oos, rets_sel)

        perf = timer.to_frame()
        print("\n--- ПРОИЗВОДИТЕЛЬНОСТЬ ---")
        print(perf.round(3).to_string(index=False))
        writer.write_performance(perf)
        writer.write_params({
            "параметры": {k: v for k, v in dataclasses.asdict(cfg).items() if k != "tickers"},
            "число_тикеров_на_входе": len(cfg.tickers),
            "период": [str(prices.index[0].date()), str(prices.index[-1].date())],
            "отобранные_активы": selected,
            "сжатие_mu_lambda": info["shrink_lambda"],
            "коррекция_PSD": info["psd_repaired"],
            "версии": package_versions(),
        })
        print("\nГотово! Все результаты сохранены.")
        return {"summary": summary, "weights": weights, "frontier": frontier,
                "frontier_cap": frontier_cap, "mc": mc,
                "oos": oos, "jumps": jumps, "selected": selected, "mu": mu, "cov": cov}

    def _plots(self, frontier, frontier_cap, mc, summary, weights, oos, rets_sel):
        plotter = Plotter(self.cfg.graphics_dir)
        try:
            plotter.frontier(frontier, frontier_cap, mc, summary, self.cfg.rf)
            for i, (name, w) in enumerate(weights.items(), start=1):
                plotter.weights(w, name, f"weights_model_{MODEL_NAMES.index(name) + 1}.png")
            plotter.metrics(summary)
            if oos is not None:
                plotter.oos(oos)
            plotter.corr_heatmap(rets_sel)
            plotter.dendrogram(rets_sel)
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Ошибка при построении графиков: {e}")
