
import polars as pl
import numpy as np
import lightgbm as lgb

from statistics import fmean
from datetime import datetime
from src.util.common import load_pickle, r2_lgb, save_as_pickle
from src.util.constants import DATA_PATH

df_validate_list = [
    pl.read_parquet(DATA_PATH / 'df_validate_1_with_features.parquet'),
    pl.read_parquet(DATA_PATH / 'df_validate_2_with_features.parquet'),
    pl.read_parquet(DATA_PATH / 'df_validate_3_with_features.parquet')
]

feature_names = ([f"feature_{i:02d}" for i in range(79)] +
                 [f'responder_{i}_mean_lag_1' for i in range(9)] +
                 ['feature_mean', 'feature_min', 'feature_max', 'feature_std'] +
                 ['time', 'weight'])

models = load_pickle(DATA_PATH / 'ensemble_models.pkl')
number_of_models = len(models)

df_prediction_list = [pl.DataFrame(), pl.DataFrame(), pl.DataFrame()]

def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, -5, 5)

# forecast with each model for each fold
for i in range(number_of_models):
    for fold in range(3):
        model = lgb.Booster(model_file=DATA_PATH / f'models/model_{i}_{fold}.txt')
        df_prediction_list[fold] = df_prediction_list[fold].with_columns(
            pl.Series(clip_predictions(model.predict(df_validate_list[fold][feature_names].to_numpy()))).alias(f'model_{i}')
        )
        print(f'{datetime.now().strftime("%H:%M:%S")} . . . Prediction for model {i}, fold {fold} done.')


def mean_r2_1_and_2(df_list):
    r2_list = []
    for fold in range(2):
        df_list[fold] = df_list[fold].with_columns(
            mean_prediction=pl.mean_horizontal(pl.all())
        )
        r2_list.append(
            r2_lgb(
                df_validate_list[fold]['responder_6'].to_numpy(),
                df_list[fold]['mean_prediction'].to_numpy(),
                sample_weight=df_validate_list[fold]['weight'].to_numpy()
            )[1]
        )
    return fmean(r2_list)


def r2_3(df_list):
    df_copy = df_list[2].with_columns(
        mean_prediction=pl.mean_horizontal(pl.all())
    )
    return r2_lgb(
        df_validate_list[2]['responder_6'].to_numpy(),
        df_copy['mean_prediction'].to_numpy(),
        sample_weight=df_validate_list[2]['weight'].to_numpy()
    )[1]


ensembled_models = []
df_prediction_ensemble_list = [pl.DataFrame(), pl.DataFrame(), pl.DataFrame()]
best_score_3 = -1
stop_counter = 0
i = 0
stopping_condition = 3

while stop_counter < stopping_condition:
    _best_score_1_and_2 = 0
    add = -1

    for i_model in range(number_of_models):
        df_prediction_ensemble_list_copy = df_prediction_ensemble_list.copy()
        for fold in range(3):
            df_prediction_ensemble_list_copy[fold] = df_prediction_ensemble_list_copy[fold].with_columns(
                model_under_test=df_prediction_list[fold][f'model_{i_model}']
            )

        _score_validation = mean_r2_1_and_2(df_prediction_ensemble_list_copy)

        if _score_validation > _best_score_1_and_2:
            _best_score_1_and_2 = _score_validation
            add = i_model

    ensembled_models.append(add)
    for fold in range(3):
        df_prediction_ensemble_list[fold] = df_prediction_ensemble_list[fold].with_columns(
            df_prediction_list[fold][f'model_{add}'].alias(f'model_{add}_{i}')
        )

    _best_score_3 = r2_3(df_prediction_ensemble_list)

    if _best_score_3 > best_score_3:
        best_score_3 = _best_score_3
        stop_counter = 0
    else:
        stop_counter = stop_counter + 1

    print(f'{datetime.now().strftime("%H:%M:%S")} . . . Added model {add}. Score on val 1 & 2: {_best_score_1_and_2}. Score on val 3: {_best_score_3}.')
    i += 1

ensembled_models = ensembled_models[:-stopping_condition]

ensembled_models_dict = {model: ensembled_models.count(model) for model in set(ensembled_models)}

save_as_pickle(ensembled_models_dict, DATA_PATH / 'ensembled_models_dict.pkl')
save_as_pickle(df_prediction_list, DATA_PATH / 'df_prediction_list.pkl')
