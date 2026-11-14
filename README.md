# Airbnb Singapore Price Prediction

Regression model that predicts the nightly price of Airbnb listings in Singapore. Submissions are ranked by Mean Absolute Percentage Error (MAPE).

## Results

| Leaderboard | MAPE | Rank |
|---|---|---|
| Public | 0.22120 | 1 / 29 |
| Private | 0.28288 | 7 / 29 |

The final model is a tuned XGBoost regressor trained on `log1p(price)`.

## Dataset

The data is split across several tables that are joined on the listing `id`.

| Table | Rows | Content |
|---|---|---|
| `train.csv` | 2,954 | Listing `id` and target `price` |
| `test.csv` | 739 | Listing `id` only |
| `airbnb.csv` | 3,693 | Listing attributes (24 columns) |
| `host.csv` | 3,693 | Host attributes (12 columns) |
| `review.csv` | 3,693 | Review score summaries (10 columns) |
| `comment.csv` | 38,350 | Individual guest comments |


## Approach

### 1. Target cleaning

- `price` is strongly right-skewed (median 232, maximum 13,000, skewness 11.42).
- 10 rows with `price <= 0` were dropped (the minimum value in the raw data is -248.75).
- 16 further rows above the 3-standard-deviation cutoff on the `log1p` scale were dropped, leaving 2,928 training rows.
- Models are trained on `log1p(price)`, and predictions are converted back with `expm1`.

### 2. Feature engineering

| Feature group | Description |
|---|---|
| Amenities | The `amenities` list is parsed and expanded into 448 binary indicators |
| Bathrooms | `bathrooms_text` is split into a bathroom count and a private/shared flag |
| Host verifications | Number of verification methods per host |
| Host rates | `host_response_rate` and `host_acceptance_rate` converted from percent strings to floats |
| Low-cardinality categoricals | Label encoding (room type, property type, response time, etc.) |
| High-cardinality categoricals | Smoothed target encoding computed out-of-fold with 5 folds (neighbourhood columns) |
| Comments | Per-listing comment count, average comment length and total comment characters |
| Text | `description`, `neighborhood_overview` and `host_about` combined, vectorized with TF-IDF (5,000 features, 1–2 grams) and reduced to 100 dimensions with truncated SVD |
| Missing values | Median imputation for numeric columns |

The final feature matrix has 590 columns: 42 tabular, 448 amenity indicators and 100 text components. TF-IDF and SVD are fitted only on the training portion of the 80/20 validation split.

### 3. Modeling

Models were compared on a held-out validation set (20% of the cleaned training data, `random_state=42`).

| Model | Validation MAPE |
|---|---|
| Baseline (mean) | 127.34% |
| Linear Regression | 53.21% |
| Random Forest | 48.08% |
| Ridge | 47.44% |
| LightGBM | 43.63% |
| Weighted ensemble (RF, XGBoost, LightGBM) | 42.47% |
| **XGBoost** | **38.95%** |

XGBoost was then tuned with randomized search (20 candidates, 3-fold cross-validation, 60 fits in total):

```python
{
    "subsample": 0.7,
    "n_estimators": 300,
    "min_child_weight": 1,
    "max_depth": 6,
    "learning_rate": 0.05,
    "colsample_bytree": 0.7,
}
```

The search reported a best cross-validated MAPE of 32.65%. The final model was refit on all 2,928 cleaned training rows and used to predict the 739 test listings.

## Repository Structure

```
.
├── hasil-5.ipynb    # Full pipeline: EDA, preprocessing, modeling, submission
├── tuning.py        # Hyperparameter search helpers used by the notebook
├── data/            
└── README.md
```

## Reproducing the Results

Requirements: Python 3, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`.

The notebook reads the competition CSV files from a `data/` directory next to it and writes `submission.csv` with the columns `id` and `price`.

## Notes

- Validation MAPE (38.95% for the untuned XGBoost) and leaderboard MAPE (0.22–0.28) are computed on different data and are not directly comparable.
- The gap between the public and private scores indicates that the model's error varies across subsets of the test listings.
