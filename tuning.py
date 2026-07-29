"""
Hyperparameter Tuning untuk XGBoost

Catatan: y yang di-fit (y_tr) adalah log1p(price). Scorer di sini
(price_mape_scorer) sudah meng-inverse ke skala price asli sebelum
menghitung MAPE, jadi best_score_ sebanding dengan MAPE yang dilaporkan
di bagian lain notebook (bukan MAPE di skala log1p).

"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
import numpy as np
import xgboost as xgb


def _mape_on_price_scale(y_true_log, y_pred_log):
    """
    FIX: y yang dipakai untuk fit (X_tr, y_tr) adalah log1p(price),
    bukan price asli. Scorer bawaan 'neg_mean_absolute_percentage_error'
    menghitung MAPE langsung di skala log1p tersebut, sehingga angka yang
    dilaporkan jauh lebih kecil dari MAPE harga sebenarnya
    (yang di validation set model lain berada di kisaran 37-47%).
    Fungsi ini meng-inverse dulu (expm1) sebelum menghitung MAPE, supaya
    hasilnya sebanding dengan MAPE yang dilaporkan di bagian lain notebook.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_absolute_percentage_error(y_true, y_pred)


price_mape_scorer = make_scorer(_mape_on_price_scale, greater_is_better=False)


def get_tuning_params():
    return {
        'n_estimators': [300, 500, 700],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
    }


def create_tuner(X, y, random_state=42, n_iter=20, cv=3, n_jobs=-1, verbose=1):
    param_dist = get_tuning_params()
    xgb_tune = xgb.XGBRegressor(random_state=random_state)
    tuner = RandomizedSearchCV(
        xgb_tune,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=price_mape_scorer,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    tuner.fit(X, y)
    return tuner


if __name__ == '__main__':
    print("File ini tidak bisa dijalankan langsung.")
    print("Import dan gunakan di notebook:")
    print("  from tuning import get_tuning_params, create_tuner")