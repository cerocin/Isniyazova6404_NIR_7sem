import platform
import time
from contextlib import contextmanager
from importlib import metadata

import pandas as pd


class StageTimer:
   
    def __init__(self):
        self.records = []
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.append((name, time.perf_counter() - t0))

    def to_frame(self):
        rows = list(self.records) + [("Итого", time.perf_counter() - self._start)]
        return pd.DataFrame(rows, columns=["Этап", "Время, с"])


def package_versions():
   
    out = {"python": platform.python_version()}
    for name in ("pandas", "numpy", "scipy", "matplotlib", "openpyxl", "requests", "apimoex"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "не установлен"
    return out
