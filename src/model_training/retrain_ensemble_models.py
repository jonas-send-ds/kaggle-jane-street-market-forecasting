
import polars as pl
import lightgbm as lgb

from statistics import fmean
from datetime import datetime
from src.util.common import load_pickle
from src.util.constants import DATA_PATH

df_all = pl.read_parquet(DATA_PATH / 'df_all_with_features.parquet')

feature_names = ([f"feature_{i:02d}" for i in range(79)] +
                 [f'responder_{i}_mean_lag_1' for i in range(9)] +
                 ['feature_mean', 'feature_min', 'feature_max', 'feature_std'] +
                 ['time', 'weight'])

models = load_pickle(DATA_PATH / 'ensemble_models.pkl')

ensembled_models_dict = load_pickle(DATA_PATH / 'ensembled_models_dict.pkl')

for i in ensembled_models_dict.keys():
    parameters = models[i][1]
    # increase training iterations by 5% (tradeoff between more data and potential overfitting on validate)
    number_trees = round(fmean(models[i][2])*1.05)

    model = lgb.LGBMRegressor(n_estimators=number_trees, objective='mse', **parameters)

    # noinspection PyTypeChecker
    model.fit(
        X=df_all[feature_names].to_numpy(),
        y=df_all['responder_6'].to_numpy(),
        sample_weight=df_all['weight'].to_numpy(),
    )

    model.booster_.save_model(DATA_PATH / f'models/model_retrained_{i}.txt')

    print(f'{datetime.now().strftime("%H:%M:%S")} . . . Re-training for model {i} done.')
