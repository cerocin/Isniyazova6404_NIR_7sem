import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.cluster import hierarchy

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]  
plt.rcParams["axes.unicode_minus"] = False

MARKERS = ["o", "s", "^", "D", "v", "P"]
COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]


class Plotter:
    def __init__(self, out_dir):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        path = self.dir / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Сохранён график: {path}")

    def frontier(self, frontier, frontier_cap, mc, summary, rf, filename="01_frontier.png"):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(mc["risk"], mc["return"], c="lightgray", s=6, alpha=0.5, label="Случайные портфели")
        ax.plot(frontier["risk"], frontier["return"], "b-", lw=2, label="Эффективная граница (без ограничения долей)")
        if frontier_cap is not None:
            ax.plot(frontier_cap["risk"], frontier_cap["return"], color="darkorange", ls="-.", lw=2,
                    label="Эффективная граница (доля актива ≤ w)")
        ok = summary.dropna(subset=["Доходность"])
        sizes = np.linspace(190, 70, len(summary))
        for i, (name, row) in enumerate(summary.iterrows()):
            if np.isnan(row["Доходность"]):
                continue
            ax.scatter(row["Риск"], row["Доходность"], s=sizes[i], marker=MARKERS[i], color=COLORS[i],
                       edgecolors="black", zorder=5 + i, label=name)
        best = ok["Шарп"].idxmax() if len(ok) and ok["Шарп"].notna().any() else None
        if best is not None and ok.loc[best, "Шарп"] > 0:
            x = ok.loc[best, "Риск"] * 1.4
            y = rf + (ok.loc[best, "Доходность"] - rf) * 1.4
            ax.plot([0, x], [rf, y], "k--", lw=1, label="Линия рынка капитала (при ставке rf)")
        ax.set_xlabel("Риск (годовое стандартное отклонение)", fontsize=11)
        ax.set_ylabel("Доходность (годовая)", fontsize=11)
        ax.set_title("Эффективная граница и портфели шести моделей", fontsize=13, pad=12)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        self._save(fig, filename)

    def weights(self, w, name, filename):
        w = w[w > 0.005].sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(w) + 1.5)))
        ax.barh(w.index, w.values, color="#1f77b4")
        for y, v in enumerate(w.values):
            ax.text(v, y, f" {v:.1%}", va="center", fontsize=8)
        ax.set_xlabel("Доля в портфеле", fontsize=11)
        ax.set_ylabel("Тикер", fontsize=11)
        ax.set_title(f"Состав портфеля: {name}", fontsize=12, pad=10)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, w.max() * 1.15)
        self._save(fig, filename)

    def metrics(self, summary, filename="metrics.png"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        specs = [("Доходность", "Годовая доходность", True), ("Риск", "Годовой риск", True),
                 ("Шарп", "Коэффициент Шарпа", False)]
        for ax, (col, title, pct) in zip(axes, specs):
            ax.barh(summary.index, summary[col].fillna(0.0), color="#1f77b4")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(title, fontsize=10)
            if pct:
                ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.invert_yaxis()
        axes[0].set_ylabel("Модель", fontsize=11)
        fig.suptitle("Сравнение моделей по метрикам (на всей выборке)", fontsize=13)
        fig.tight_layout()
        self._save(fig, filename)

    def oos(self, oos, filename="oos_sharpe.png"):
        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(oos))
        ax.barh(y - 0.2, oos["Шарп (обучение)"].fillna(0.0), height=0.4, label="Обучающая выборка")
        ax.barh(y + 0.2, oos["Шарп (тест)"].fillna(0.0), height=0.4, label="Тестовая выборка")
        ax.set_yticks(y)
        ax.set_yticklabels(oos.index)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Коэффициент Шарпа", fontsize=11)
        ax.set_ylabel("Модель", fontsize=11)
        ax.set_title("Проверка на отложенной выборке: коэффициент Шарпа", fontsize=12, pad=10)
        ax.legend()
        self._save(fig, filename)

    def corr_heatmap(self, rets, filename="corr_heatmap.png"):
        corr = rets.corr(min_periods=30)
        n = len(corr)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_xlabel("Тикер", fontsize=11)
        ax.set_ylabel("Тикер", fontsize=11)
        ax.set_title("Корреляционная матрица доходностей", fontsize=14, pad=14)
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    v = corr.iat[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Коэффициент корреляции")
        self._save(fig, filename)

    def dendrogram(self, rets, filename="dendrogram.png"):
        corr = rets.corr(min_periods=30).fillna(0.0).to_numpy()
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(dist, 0.0)
        iu = np.triu_indices(len(dist), 1)
        link = hierarchy.linkage(dist[iu], method="average")
        fig, ax = plt.subplots(figsize=(13, 6))
        hierarchy.dendrogram(link, labels=list(rets.columns), leaf_rotation=90, leaf_font_size=8, ax=ax)
        ax.set_xlabel("Тикер", fontsize=11)
        ax.set_ylabel("Расстояние d = √(0,5·(1 − ρ))", fontsize=11)
        ax.set_title("Дендрограмма иерархической кластеризации активов", fontsize=13, pad=12)
        self._save(fig, filename)
