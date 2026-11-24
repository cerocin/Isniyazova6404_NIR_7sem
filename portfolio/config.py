from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_TICKERS = [
    "ROSN", "GAZP", "SBER", "TATN", "VTBR", "GMKN", "CHMF", "MAGN",
    "NLMK", "BANE", "TRNFP", "ALRS", "SMLT", "EUTR", "NVTK", "LKOH",
    "RUAL", "PLZL", "SELG", "UGLD", "MOEX", "SPBE", "BSPB", "SFIN",
    "IVAT", "DIAS", "SOFL", "BELU", "APTK", "AQUA", "DATA", "DOMRF", "ENPG",
    "FESH", "FEES", "ETLN", "FLOT", "GAZA", "GECO", "GEMA", "GLRX", "HNFG", "HYDR",
    "KLVZ", "KMAZ", "KROT", "LEAS", "LSRG", "MDMG", "MTLR", "MVID", "OZPH", "PHOR",
    "PRMD", "RAGR", "RASP", "RENI", "RTKM", "SIBN", "SGZH", "SNGS", "SVCB", "UGLD",
    "UPRO", "IRAO", "UTAR", "VSEH", "WUSH",
]
# убираем повторы (UGLD был в списке дважды), порядок сохраняется
DEFAULT_TICKERS = list(dict.fromkeys(_TICKERS))


@dataclass
class Config:
    tickers: list = field(default_factory=lambda: list(DEFAULT_TICKERS))
    start: str = None
    end: str = None
    input_csv: Path = None
    data_dir: Path = ROOT / "data"
    results_dir: Path = ROOT / "results"
    graphics_dir: Path = ROOT / "graphics"
    # параметры моделей
    max_risk: float = 0.30
    target_return: float = 0.20
    max_weight: float = 0.30
    rf: float = 0.15
    # параметры обработки данных
    trading_days: int = 252
    min_coverage: float = 0.75
    min_obs: int = 120
    min_pair_obs: int = 60
    max_gap_days: int = 10
    jump_threshold: float = 0.4
    min_assets: int = 3
    min_span_days: int = 365
    shrink_mu: bool = True
    holdout: float = 0.3
    mc_samples: int = 3000
    frontier_points: int = 30
    seed: int = 42

    def validate(self):
        errors = []
        today = datetime.now().date()
        dates = {}
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                d = datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"Неверный формат --{name}: «{value}», нужен YYYY-MM-DD.")
                continue
            if d > today:
                errors.append(f"Дата --{name} не может быть в будущем.")
            dates[name] = d
        if "start" in dates and "end" in dates:
            if dates["start"] >= dates["end"]:
                errors.append("Дата начала должна быть раньше даты конца.")
            elif (dates["end"] - dates["start"]).days < self.min_span_days:
                errors.append(f"Период короче {self.min_span_days} дней.")
        if not 0 < self.max_risk <= 1:
            errors.append(f"--max-risk={self.max_risk} вне диапазона (0, 1].")
        if not 0 < self.target_return <= 2:
            errors.append(f"--target-return={self.target_return} вне диапазона (0, 2].")
        if not 0.01 <= self.max_weight <= 1:
            errors.append(f"--max-weight={self.max_weight} вне диапазона [0.01, 1].")
        if not 0 <= self.rf < 1:
            errors.append(f"--rf={self.rf} вне диапазона [0, 1).")
        if not 0 <= self.holdout < 0.8:
            errors.append(f"--holdout={self.holdout} вне диапазона [0, 0.8).")
        if not 0 < self.min_coverage <= 1:
            errors.append(f"--min-coverage={self.min_coverage} вне диапазона (0, 1].")
        if self.jump_threshold <= 0 or self.max_gap_days < 1 or self.trading_days < 1:
            errors.append("jump_threshold, max_gap_days и trading_days должны быть положительными.")
        if self.mc_samples < 100 or self.frontier_points < 5:
            errors.append("mc_samples >= 100 и frontier_points >= 5.")
        if not self.tickers:
            errors.append("Список тикеров пуст.")
        if errors:
            raise ValueError("Некорректные параметры:\n  - " + "\n  - ".join(errors))
