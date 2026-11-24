import time
from datetime import datetime, timedelta
from pathlib import Path
import urllib3

urllib3.disable_warnings()

import pandas as pd
import requests

ISS_URL = "https://iss.moex.com/iss/index.json"


class MoexLoader:
    # загрузка дневных цен закрытия с MOEX ISS
    def __init__(self, cfg, pause=0.2, retries=2):
        self.cfg = cfg
        self.pause = pause
        self.retries = retries

    def check_connection(self):
        try:
            return requests.get(ISS_URL, timeout=5, verify=False).status_code == 200
        except requests.RequestException:
            return False

    def load_ticker(self, session, ticker, start, end):
        try:
            import apimoex
        except ImportError as e:
            raise ConnectionError("Библиотека apimoex не установлена (pip install apimoex).") from e
        rows = None
        for attempt in range(self.retries + 1):
            try:
                session.verify = False
                rows = apimoex.get_board_history(session, ticker, start=start, end=end)
                break
            except requests.RequestException as e:
                print(f"  [СЕТЕВАЯ ОШИБКА] {ticker}, попытка {attempt + 1}: {e}")
                time.sleep(1.0)
        if not rows:
            return None
        s = pd.DataFrame(rows).set_index("TRADEDATE")["CLOSE"]
        s = s[~s.index.duplicated(keep="last")]
        s.name = ticker
        return s

    def load_all(self):
        if not self.check_connection():
            raise ConnectionError("MOEX ISS недоступен. Проверьте интернет.")
        end = self.cfg.end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start = self.cfg.start or (datetime.now() - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
        print(f"Период загрузки: с {start} по {end}")
        series, failed = [], []
        with requests.Session() as session:
            for t in self.cfg.tickers:
                s = self.load_ticker(session, t, start, end)
                if s is None:
                    failed.append(t)
                else:
                    series.append(s)
                    print(f"  {t}: {len(s)} записей")
                time.sleep(self.pause)
        print(f"[ЗАГРУЗКА ЗАВЕРШЕНА] успешно: {len(series)}, не удалось: {len(failed)} {failed}")
        if not series:
            raise ValueError("Не удалось загрузить данные ни по одному тикеру.")
        return pd.concat(series, axis=1).sort_index()


class CsvStore:
    # чтение и запись таблицы цен в CSV (кэш и ручной ввод данных)
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        if not self.path.is_file():
            raise FileNotFoundError(f"Файл с ценами не найден: {self.path}")
        df = pd.read_csv(self.path, index_col=0)
        if df.empty:
            raise ValueError(f"Файл {self.path} не содержит данных.")
        return df

    def write(self, prices):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(self.path)
        print(f"Сохранено: {self.path}, форма: {prices.shape}")
