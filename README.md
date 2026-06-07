# Tokyo Airbnb Price Analysis

Exploratory data analysis and price prediction for Tokyo Airbnb listings using 2025 [Inside Airbnb](http://insideairbnb.com/) data.

## Key Findings

- Median Tokyo Airbnb price: **~¥16,200/night** (~$108 USD)
- Entire home/apt dominates at **85% of listings**
- Central wards (Minato, Chuo, Shinjuku) command the highest prices
- Prices spike in **April** — cherry blossom season and Golden Week
- Ridge regression achieves R²=0.13, MAE=¥8,583 — room type and neighbourhood alone explain ~10% of price variance

## Structure

```
airbnb-tokyo/
├── data/
│   └── raw/
│       └── listings.csv        # Inside Airbnb Tokyo dataset (27,945 listings, 18 features)
├── notebooks/
│   └── airbnb_tokyo_analysis.ipynb
└── outputs/                    # Generated charts (gitignored if empty)
```

## Notebook Outline

1. **EDA** — price distribution, room type breakdown, top neighbourhoods by median price, monthly price trends
2. **ML** — Ridge regression with one-hot encoded room type and neighbourhood; actual vs. predicted scatter plot

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook notebooks/airbnb_tokyo_analysis.ipynb
```

## Limitations & Next Steps

- `last_review` date used as a booking date proxy — not actual booking date
- No amenity, host quality, or photo data in this dataset
- Log-transforming price or switching to a tree-based model (Random Forest, XGBoost) would likely improve predictions on high-end listings
