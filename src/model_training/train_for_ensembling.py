
import polars as pl
import lightgbm as lgb

from statistics import fmean
from datetime import datetime
from src.util.common import load_pickle, save_as_pickle
from src.util.constants import DATA_PATH


feature_names = ([f"feature_{i:02d}" for i in range(79)] +
                 [f'responder_{i}_mean_lag_1' for i in range(9)] +
                 ['feature_mean', 'feature_min', 'feature_max', 'feature_std'] +
                 ['time', 'weight'])

models = load_pickle(DATA_PATH / 'models.pkl')

# sort models by average r2
for i in range(len(models)):
    r2_mean = fmean(models[i][0])
    models[i].append(r2_mean)
models.sort(key=lambda x: x[3], reverse=True)

# train and save the top 10 models
for i in range(10):
    parameters = models[i][1]
    for j in range(3):
        if j < 2:
            number_trees = models[i][2][j]
        else:
            number_trees = round(fmean(models[i][2]))

        model = lgb.LGBMRegressor(n_estimators=number_trees, objective='mse', **parameters)

        df_train = pl.read_parquet(DATA_PATH / f'df_train_{j+1}_with_features.parquet')

        # noinspection PyTypeChecker
        model.fit(
            X=df_train[feature_names].to_numpy(),
            y=df_train['responder_6'].to_numpy(),
            sample_weight=df_train['weight'].to_numpy(),
        )

        model.booster_.save_model(DATA_PATH / f'models/model_{i}_{j}.txt')

        print(f'{datetime.now().strftime("%H:%M:%S")} . . . Training for model {i}, fold {j} done.')

save_as_pickle(models[0:10], DATA_PATH / 'ensemble_models.pkl')
