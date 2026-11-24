# Оптимизация портфеля акций MOEX (НИР)

## Запуск
```
pip install -r requirements.txt
python main.py                                   # загрузка с MOEX ISS, период по умолчанию — 4 года
python main.py --start 2023-01-01 --end 2025-12-31 --max-weight 0.25 --rf 0.16
python main.py --input data/raw_prices.csv       # расчёт по готовой таблице цен (без интернета)
python -m unittest discover tests                # тесты
```
Параметры: `python main.py --help`. Новый временной срез задаётся `--start/--end`, состав бумаг — `--tickers-file`.

## Структура
| Модуль | Класс | Назначение |
|---|---|---|
| `portfolio/config.py` | `Config` | параметры запуска и их проверка |
| `portfolio/data_loader.py` | `MoexLoader`, `CsvStore` | загрузка MOEX ISS; чтение и запись CSV |
| `portfolio/validator.py` | `PriceValidator`, `ValidationReport` | проверка таблицы цен |
| `portfolio/preprocessing.py` | `ReturnsBuilder`, `MomentsEstimator` | доходности, отбор активов, mu и Sigma |
| `portfolio/optimizer.py` | `PortfolioOptimizer`, `InfeasibleError` | 6 моделей, граница, метрики |
| `portfolio/simulation.py` | `RandomPortfolios` | случайные портфели, фронт Парето |
| `portfolio/plotting.py` | `Plotter` | графики (подписи на русском) |
| `portfolio/report.py` | `ReportWriter` | сохранение CSV, XLSX, JSON |
| `portfolio/pipeline.py` | `PortfolioPipeline` | полный цикл расчёта |
| `portfolio/utils.py` | `StageTimer` | замер времени этапов, версии библиотек |

## Файлы
Вход: CSV, первый столбец — дата `YYYY-MM-DD`, остальные — тикеры, значения — цены закрытия (пропуски допустимы).
Выход (`results/`): `summary.csv`, `weights.csv`, `results.xlsx` (итоги, веса, границы, Монте-Карло, Парето, скачки цен, проверка данных, вне выборки), `performance.csv`, `run_params.json`.
Графики (`graphics/`): граница, веса шести моделей, метрики, отложенная выборка, корреляции, дендрограмма.

## Исключения
| Ситуация | Исключение | Код выхода |
|---|---|---|
| нет файла цен или списка тикеров | `FileNotFoundError` | 2 |
| неверные параметры, пустая или короткая выборка, мало активов | `ValueError` | 3 |
| нулевой риск портфеля при расчёте Шарпа | `ZeroDivisionError` | 4 |
| MOEX недоступен и нет локальной копии | `ConnectionError` | 5 |
| недопустимая модель (R, r, w, rf) | `InfeasibleError(ValueError)` — модель помечается «нет решения», расчёт продолжается | — |

## Сложность (T — дней, N — активов, M — случайных портфелей, P — точек границы, k — итераций SLSQP)
Проверка и доходности O(T·N); ковариация O(T·N²); коррекция PSD O(N³); одна модель O(k·N³); граница O(P·k·N³); Монте-Карло O(M·N²); фронт Парето O(M·log M).
Замер (T=1241, N=50, без сети): оптимизация 6 моделей 0,07 с, граница и Монте-Карло 0,7 с, весь расчёт 3,6 с (из них графики 2,6 с).

## Допущения и ограничения
- Цены закрытия не включают дивиденды: это ценовая, а не полная доходность.
- Сплиты и консолидации исключаются из выборки по порогу скачка `--jump-threshold`; список событий лежит на листе «Скачки цен» и требует ручной проверки.
- Безрисковая ставка `--rf` по умолчанию 15 %: задайте среднюю ключевую ставку ЦБ или доходность ОФЗ за период.
- Короткие продажи запрещены (w ≥ 0).
