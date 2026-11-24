import json

import pandas as pd


class ReportWriter:
    # сохранение всех результатов расчёта
    def __init__(self, results_dir):
        self.dir = results_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_weights(self, weights, filename="weights.csv"):
        path = self.dir / filename
        pd.DataFrame(weights).to_csv(path, encoding="utf-8-sig")
        print(f"Сохранено: {path}")

    def write_summary(self, summary, filename="summary.csv"):
        path = self.dir / filename
        summary.to_csv(path, encoding="utf-8-sig")
        print(f"Сохранено: {path}")

    def write_excel(self, tables, filename="results.xlsx"):
        # tables: {имя листа: DataFrame, index_flag}
        path = self.dir / filename
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet, (df, keep_index) in tables.items():
                df.to_excel(writer, sheet_name=sheet, index=keep_index)
        print(f"Сохранено: {path}")

    def write_performance(self, perf, filename="performance.csv"):
        path = self.dir / filename
        perf.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Сохранено: {path}")

    def write_params(self, params, filename="run_params.json"):
        path = self.dir / filename
        path.write_text(json.dumps(params, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"Сохранено: {path}")
