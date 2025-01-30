
import polars as pl
import optuna
import lightgbm as lgb

from lightgbm import early_stopping, log_evaluation

from src.util.common import r2_lgb
from src.util.constants import DATA_PATH

df_train = pl.read_parquet(DATA_PATH / 'df_train.parquet')
df_validate = pl.read_parquet(DATA_PATH / 'df_validate.parquet')

feature_names = [f"feature_{i:02d}" for i in range(79)]

fixed_parameters = {
    "learning_rate": .1,
    "n_jobs": 12,  # current number of cores for Mac
    "subsample_freq": 1,
    "verbose": -1,
    "metric": "None"
}


def objective(trial):
    parameter_sample = {
        "learning_rate": trial.suggest_float("learning_rate", .01, .5, log=True),
        'max_bin': trial.suggest_int('max_bin', 2 ** 3 - 1, 2 ** 10 - 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 16),  # 3 < x
        "num_leaves": trial.suggest_int("num_leaves", 2 ** 3 - 1, 2 ** 12 - 1, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", .1, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", .1, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 1000000, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0000001, 100000, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0000001, 100000, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0000001, 100000, log=True),
    }

    parameters = {**fixed_parameters, **parameter_sample}

    lgb_callbacks = [
        early_stopping(stopping_rounds=10, first_metric_only=True, verbose=False),
        log_evaluation(100)
    ]

    lgb_eval_set = [(
        df_validate[feature_names].to_numpy(),
        df_validate['responder_6'].to_numpy()
    )]

    model = lgb.LGBMRegressor(n_estimators=round(50 / parameters['learning_rate']), objective='mse', **parameters)

    # noinspection PyTypeChecker
    model.fit(
        X=df_train[feature_names].to_numpy(),
        y=df_train['responder_6'].to_numpy(),
        sample_weight=df_train['weight'].to_numpy(),
        eval_metric=[r2_lgb],
        eval_set=lgb_eval_set,
        eval_sample_weight=[df_validate['weight'].to_numpy()],
        callbacks=lgb_callbacks
    )

    return model.best_score_['valid_0']['r2']


# optimise the objective function
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# visualise result
for param in ['learning_rate', 'max_bin', 'max_depth', 'num_leaves', 'bagging_fraction', 'feature_fraction', 'min_data_in_leaf', 'min_sum_hessian_in_leaf', 'lambda_l1', 'lambda_l2']:
    optuna.visualization.plot_slice(study, params=[param]).show()

