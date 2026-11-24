from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    rows_in: int = 0
    cols_in: int = 0
    bad_dates: int = 0
    duplicate_dates: int = 0
    text_cells: int = 0
    nonpositive_cells: int = 0
    nan_cells: int = 0
    empty_columns: list = field(default_factory=list)
    rows_out: int = 0
    cols_out: int = 0

    def to_frame(self):
        items = [
            ("Строк на входе", self.rows_in),
            ("Столбцов на входе", self.cols_in),
            ("Некорректных дат (строки удалены)", self.bad_dates),
            ("Повторяющихся дат (строки удалены)", self.duplicate_dates),
            ("Текстовых ячеек (заменены на NaN)", self.text_cells),
            ("Цен <= 0 (заменены на NaN)", self.nonpositive_cells),
            ("Пустых столбцов (удалены)", ", ".join(self.empty_columns) or "нет"),
            ("Ячеек с пропуском после очистки", self.nan_cells),
            ("Строк на выходе", self.rows_out),
            ("Столбцов на выходе", self.cols_out),
        ]
        return pd.DataFrame(items, columns=["Показатель", "Значение"])


class PriceValidator:
    # собственная проверка таблицы цен: даты, текст, нули, пропуски
    def __init__(self, cfg):
        self.cfg = cfg

    def validate(self, prices):
        if prices is None or prices.empty:
            raise ValueError("Таблица цен пуста.")
        rep = ValidationReport(rows_in=prices.shape[0], cols_in=prices.shape[1])
        df = prices.copy()

        # 1. индекс -> даты, удаление битых и повторных дат
        idx = pd.to_datetime(df.index, errors="coerce")
        bad = np.asarray(idx.isna())
        rep.bad_dates = int(bad.sum())
        df = df.loc[~bad]
        df.index = idx[~bad]
        dup = df.index.duplicated(keep="last")
        rep.duplicate_dates = int(dup.sum())
        df = df.loc[~dup].sort_index()

        # 2. текст в числовых ячейках
        num = df.apply(pd.to_numeric, errors="coerce")
        text = df.notna() & num.isna()
        rep.text_cells = int(text.to_numpy().sum())
        for col in df.columns[text.any().to_numpy()]:
            example = df.loc[text[col], col].iloc[0]
            print(f"  [!] В столбце {col} найден текст (пример: {example!r}); заменён на NaN.")
        df = num.astype(float).replace([np.inf, -np.inf], np.nan)

        # 3. цены должны быть строго положительными
        nonpos = df <= 0
        rep.nonpositive_cells = int(nonpos.to_numpy().sum())
        if rep.nonpositive_cells:
            print(f"  [!] Найдено {rep.nonpositive_cells} цен <= 0; заменены на NaN.")
        df = df.mask(nonpos)

        # 4. полностью пустые столбцы
        empty = df.columns[df.isna().all().to_numpy()]
        rep.empty_columns = list(empty)
        df = df.drop(columns=empty)
        if df.empty:
            raise ValueError("После проверки в таблице не осталось данных.")

        rep.nan_cells = int(df.isna().to_numpy().sum())
        rep.rows_out, rep.cols_out = df.shape
        print(f"[ПРОВЕРКА ДАННЫХ] {rep.rows_in}x{rep.cols_in} -> {rep.rows_out}x{rep.cols_out}, "
              f"текст: {rep.text_cells}, цен<=0: {rep.nonpositive_cells}, "
              f"дублей дат: {rep.duplicate_dates}, пропусков: {rep.nan_cells}")
        return df, rep
