# QuickCart Stockout Risk

Predicting whether a product at a store is **Safe**, **At-Risk**, or **Imminent** (about to run out before the next delivery) — a daily, per-store, per-SKU classification problem for a quick-commerce warehouse operator.

## The data

- 12 dark stores across 6 Indian metros
- 60 SKUs across 8 categories
- 15 suppliers
- 30 days of daily inventory data (Oct 1–30, 2026)
- ~21,600 rows in the main modeling table, joined from 5 CSVs (stores, SKUs, suppliers, events, and the daily fact table)

Target distribution: Safe 65.4% / At-Risk 24.0% / Imminent 10.6% — a realistic operational imbalance, similar to fraud detection or ER triage.

## What's in this repo

- `QuickCart_Stockout_Risk.ipynb` — full notebook: data loading, two data-quality fixes, EDA, feature engineering, a time-based train/test split, and three models (baseline, Logistic Regression, Random Forest)
- `dim_stores.csv`, `dim_skus.csv`, `dim_suppliers.csv`, `dim_events.csv`, `fact_inventory_daily.csv` — the raw data
- `requirements.txt` — Python dependencies

## Approach

- **Two data-quality issues fixed on load**: inconsistent city casing in `dim_stores`, and a `reliability_score` column where missing values were stored as the literal string `'N/A'`.
- **Time-based split** (train on Oct 1–23, test on Oct 24–30) instead of a random split, since the data is a daily panel and a random split would leak store-SKU patterns across train and test.
- **Class-weighted models**, since Imminent is only ~10% of the data and accuracy alone would reward a model that never predicts it.
- **Evaluation focused on Imminent recall** — missing a real stockout is more costly than a false alarm.

## Results

| Model | Accuracy | Imminent Recall |
|---|---|---|
| Baseline (majority class) | ~62% | 0% |
| Logistic Regression | ~90% | ~88% |
| Random Forest | ~92% | ~81% |

`days_of_cover_ratio` and `reorder_gap` were the strongest predictors — the two most direct numeric signals of a product about to run out.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook QuickCart_Stockout_Risk.ipynb
```

## Author

[Dvs](https://github.com/Devesh-Vishwasrao-20) — Data Science and Analytics, The Unlox Academy
