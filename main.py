import argparse
import sys
from pathlib import Path

from portfolio.config import DEFAULT_TICKERS, Config
from portfolio.pipeline import PortfolioPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Оптимизация портфеля акций MOEX (6 моделей)")
    p.add_argument("--input", type=Path, help="CSV с ценами (если не задан — загрузка с MOEX ISS)")
    p.add_argument("--tickers-file", type=Path, help="файл со списком тикеров (по одному в строке)")
    p.add_argument("--start", help="дата начала YYYY-MM-DD")
    p.add_argument("--end", help="дата конца YYYY-MM-DD")
    p.add_argument("--max-risk", type=float, default=0.30, help="макс. риск R (годовое СКО)")
    p.add_argument("--target-return", type=float, default=0.20, help="целевая доходность r")
    p.add_argument("--max-weight", type=float, default=0.30, help="макс. доля актива w")
    p.add_argument("--rf", type=float, default=0.15, help="безрисковая ставка (годовая)")
    p.add_argument("--holdout", type=float, default=0.3, help="доля отложенной выборки (0 — выкл.)")
    p.add_argument("--no-shrink", action="store_true", help="не сжимать оценки mu")
    p.add_argument("--min-coverage", type=float, default=0.75, help="мин. доля наблюдений актива")
    p.add_argument("--max-gap-days", type=int, default=10, help="макс. разрыв календаря, дн.")
    p.add_argument("--jump-threshold", type=float, default=0.4, help="порог скачка |ln(P1/P0)|")
    p.add_argument("--mc-samples", type=int, default=3000, help="число случайных портфелей")
    p.add_argument("--seed", type=int, default=42, help="seed генератора случайных чисел")
    return p.parse_args()


def read_tickers(path):
    if not path.is_file():
        raise FileNotFoundError(f"Файл с тикерами не найден: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    tickers = [ln.upper() for ln in lines if ln and not ln.startswith("#")]
    if not tickers:
        raise ValueError(f"Файл {path} не содержит тикеров.")
    return list(dict.fromkeys(tickers))


def build_config(a):
    tickers = read_tickers(a.tickers_file) if a.tickers_file else list(DEFAULT_TICKERS)
    return Config(tickers=tickers, start=a.start, end=a.end, input_csv=a.input,
                  max_risk=a.max_risk, target_return=a.target_return, max_weight=a.max_weight,
                  rf=a.rf, holdout=a.holdout, shrink_mu=not a.no_shrink,
                  min_coverage=a.min_coverage, max_gap_days=a.max_gap_days,
                  jump_threshold=a.jump_threshold, mc_samples=a.mc_samples, seed=a.seed)


def main():
    try:
        cfg = build_config(parse_args())
        cfg.validate()
        PortfolioPipeline(cfg).run()
    except FileNotFoundError as e:
        print(f"\n[ОШИБКА: файл не найден] {e}")
        return 2
    except ConnectionError as e:
        print(f"\n[ОШИБКА: нет связи с MOEX ISS] {e}")
        return 5
    except ZeroDivisionError as e:
        print(f"\n[ОШИБКА: деление на ноль] {e}")
        return 4
    except ValueError as e:
        print(f"\n[ОШИБКА: некорректные данные или параметры] {e}")
        return 3
    except KeyboardInterrupt:
        print("\n[ОСТАНОВЛЕНО] Пользователь прервал выполнение.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
