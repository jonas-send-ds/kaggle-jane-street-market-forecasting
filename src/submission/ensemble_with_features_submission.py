
import kaggle_evaluation.jane_street_inference_server
import lightgbm as lgb
import polars as pl
import numpy as np
import os
import pickle

from typing import Any
from pathlib import Path

lags_ : pl.DataFrame | None = None
df_lags : pl.DataFrame | None = None
max_time_id = 967


def load_pickle(path: str | Path) -> Any:
    open_file = open(path, "rb")
    x = pickle.load(open_file)
    open_file.close()
    return x


feature_names = ([f"feature_{i:02d}" for i in range(79)] +
                 [f'responder_{i}_mean_lag_1' for i in range(9)] +
                 ['feature_mean', 'feature_min', 'feature_max', 'feature_std'] +
                 ['time', 'weight'])
feature_columns = [f'feature_{i:02}' for i in range(79) if i not in [9, 10, 11]]  # exclude the integer columns

ensembled_models_dict = load_pickle('/kaggle/input/ensembled-models-dict/ensembled_models_dict.pkl')
models = {}
for i in ensembled_models_dict.keys():
    models[i] = lgb.Booster(model_file=f'/kaggle/input/retrained_models_with_features/other/default/1/model_retrained_{i}.txt')


def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, -5, 5)


def create_lag_features(_lags):
    for i in range(9):
        _lags = _lags.with_columns(
            pl.col(f'responder_{i}_lag_1').mean().over('symbol_id').alias(f'responder_{i}_mean_lag_1')
        )

    _lags = _lags.select(['symbol_id'] + [f'responder_{i}_mean_lag_1' for i in range(9)])
    _lags = _lags.unique(maintain_order=True)

    return _lags


def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    global lags_, models, ensembled_models_dict, feature_names, df_lags, max_time_id

    if lags is not None:
        lags_ = lags
        df_lags = create_lag_features(lags_)
        max_time_id = lags_['time_id'].max()

    test = test.with_columns(
        time=pl.col('time_id') / max_time_id,
        feature_mean = pl.mean_horizontal(feature_columns),
        feature_min = pl.min_horizontal(feature_columns),
        feature_max = pl.max_horizontal(feature_columns),
        feature_std = pl.concat_list(feature_columns).list.std()
    )

    test = test.join(df_lags, on='symbol_id', how='left')

    df_predictions = pl.DataFrame()
    for i in models.keys():
        predictions = clip_predictions(models[i].predict(test[feature_names].to_numpy()))
        for j in range(ensembled_models_dict[i]):
            df_predictions = df_predictions.with_columns(
                pl.Series(predictions).alias(f"model_{i}_{j}")
            )

    df_predictions = df_predictions.with_columns(
        mean_prediction=pl.mean_horizontal(pl.all())
    )

    predictions = test.select('row_id').with_columns(
            pl.Series(
                name='responder_6',
                values=df_predictions['mean_prediction'],
                dtype=pl.Float64,
            )
        )

    assert isinstance(predictions, pl.DataFrame)
    assert predictions.columns == ['row_id', 'responder_6']
    assert len(predictions) == len(test)
    return predictions

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )
