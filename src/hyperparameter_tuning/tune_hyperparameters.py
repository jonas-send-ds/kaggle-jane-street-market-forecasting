
import polars as pl
import lightgbm as lgb
import numpy as np
import optuna

from lightgbm import early_stopping, log_evaluation
from src.util.common import r2_lgb, save_as_pickle, load_pickle
from src.util.constants import DATA_PATH


df_train_list = [
    pl.read_parquet(DATA_PATH / 'df_train_1_with_features.parquet'),
    pl.read_parquet(DATA_PATH / 'df_train_2_with_features.parquet')
]
df_validate_list = [
    pl.read_parquet(DATA_PATH / 'df_validate_1_with_features.parquet'),
    pl.read_parquet(DATA_PATH / 'df_validate_2_with_features.parquet')
]

feature_names = ([f"feature_{i:02d}" for i in range(79)] +
                 [f'responder_{i}_mean_lag_1' for i in range(9)] +
                 ['feature_mean', 'feature_min', 'feature_max', 'feature_std'] +
                 ['time', 'weight'])

fixed_parameters = {
    "learning_rate": .1,
    "n_jobs": 12,  # current number of cores for Mac
    "subsample_freq": 1,
    "verbose": -1,
    "metric": "None"
}

models = []  # or models = load_pickle(DATA_PATH / 'models.pkl')


def objective(trial):
    parameter_sample = {
        "learning_rate": trial.suggest_float("learning_rate", .005, .025, log=True),
        'max_bin': trial.suggest_int('max_bin', 2 ** 5 - 1, 2 ** 8 - 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 9, 16),
        "num_leaves": trial.suggest_int("num_leaves", 2 ** 10 - 1, 2 ** 13 - 1, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", .3, .7),
        "feature_fraction": trial.suggest_float("feature_fraction", .3, .7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 2000, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.001, 100, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0000000001, .00001, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 100, 10000, log=True)
    }

    print(f'starting trial with parameters: {parameter_sample}')

    parameters = {**fixed_parameters, **parameter_sample}

    result_r2 = []
    result_best_iteration = []

    for fold in range(2):
        lgb_callbacks = [
            early_stopping(stopping_rounds=10, first_metric_only=True, verbose=False),
            log_evaluation(100)
        ]

        lgb_eval_set = [(
            df_validate_list[fold][feature_names].to_numpy(),
            df_validate_list[fold]['responder_6'].to_numpy()
        )]

        model = lgb.LGBMRegressor(n_estimators=round(50 / parameters['learning_rate']), objective='mse', **parameters)

        # noinspection PyTypeChecker
        model.fit(
            X=df_train_list[fold][feature_names].to_numpy(),
            y=df_train_list[fold]['responder_6'].to_numpy(),
            sample_weight=df_train_list[fold]['weight'].to_numpy(),
            eval_metric=[r2_lgb],
            eval_set=lgb_eval_set,
            eval_sample_weight=[df_validate_list[fold]['weight'].to_numpy()],
            callbacks=lgb_callbacks
        )

        result_r2.append(model.best_score_['valid_0']['r2'])
        result_best_iteration.append(model.best_iteration_)

        model.booster_.save_model(DATA_PATH / 'models/tmp/model_optuna_tmp.txt')


    models.append([result_r2, parameters, result_best_iteration])
    save_as_pickle(models, DATA_PATH / 'models_backup.pkl')  # back up results
    return np.mean(result_r2)


# optimise the objective function
study = optuna.create_study(direction="maximize")  # or study = load_pickle(DATA_PATH / 'study.pkl')
study.optimize(objective, n_trials=100, show_progress_bar=True)

# visualise result
for param in ['learning_rate', 'max_bin', 'max_depth', 'num_leaves', 'bagging_fraction', 'feature_fraction', 'min_data_in_leaf', 'min_sum_hessian_in_leaf', 'lambda_l1', 'lambda_l2']:
    optuna.visualization.plot_slice(study, params=[param]).show()

# save_as_pickle(study, DATA_PATH / 'study.pkl')
# save_as_pickle(models, DATA_PATH / 'models.pkl')

# check correlation between val 1 and 2
r2_list = models.copy()
for model_list in r2_list:
    if len(model_list) > 2:
        model_list.pop(2)
    model_list.pop(1)

# Flatten each sublist within the list to ensure no inner lists remain
r2_list_flattened = [
    [item for sublist in model_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    for model_list in r2_list]

# Create a Polars dataframe with R2 scores
df_r2 = pl.DataFrame({
    "r2_1": [model_list[0] for model_list in r2_list_flattened],
    "r2_2": [model_list[1] for model_list in r2_list_flattened]
})

corr = df_r2.corr()[0, 1]
print(corr)  # 85+ %
