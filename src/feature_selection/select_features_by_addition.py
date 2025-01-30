
import polars as pl
import lightgbm as lgb
import src.util.common as common

from lightgbm import early_stopping, log_evaluation
from datetime import datetime
from src.util.common import r2_lgb, save_as_pickle, load_pickle
from src.util.constants import DATA_PATH


NUMBER_OF_MODELS = 7

base_features = ([f"feature_{i:02d}" for i in range(79)] +
                 [f'responder_{i}_mean_lag_1' for i in range(9)] +
                 ['feature_mean', 'feature_min', 'feature_max', 'feature_std'] +
                 ['time'])
feature_sets = [
    [],
    [f'responder_{i}_std_lag_1' for i in range(9)],
    ['weight']
]

fixed_parameters = {
    "learning_rate": .1,
    "n_jobs": 12,  # current number of cores for Mac
    "subsample_freq": 1,
    "verbose": -1,
    "metric": "None"
}

hyperparameter_space = {
    "learning_rate": [.025],
    'max_bin': [2 ** x - 1 for x in list(range(5, 9))],
    "max_depth": list(range(9, 13)),
    "num_leaves": [2 ** x - 1 for x in list(range(7, 12))],
    "bagging_fraction": [.1, .2, .3, .4, .5, .6, .7],
    "feature_fraction": [.2, .3, .4, .5, .6, .7],
    "min_data": [50, 100, 200, 500, 1000, 2000],
    "min_sum_hessian_in_leaf": [.001, .005, .01, .1, 1, 10, 100, 1000],
    "lambda_l1": [0, .00001, .0001, .001, .005, .01],
    "lambda_l2": [100, 1000, 10000, 50000]
}

result = []
result_feature_importance = []
# result = load_pickle(DATA_PATH / "feature_addition_result.pkl")

for i in range(NUMBER_OF_MODELS):
# for i in range(1):
    sampled_parameters = common.sample_parameters(hyperparameter_space)
    parameters = {**fixed_parameters, **sampled_parameters}

    result_model = []
    result_feature_importance_model = []

    for j in range(len(feature_sets)):
        result_feature_set = []
        result_feature_importance_feature_set = []

        for fold in range(2):
            df_train = pl.read_parquet(DATA_PATH / f'df_train_{fold+1}_with_features.parquet')
            df_validate = pl.read_parquet(DATA_PATH / f'df_validate_{fold+1}_with_features.parquet')

            lgb_callbacks = [
                early_stopping(stopping_rounds=10, first_metric_only=True, verbose=False),
                log_evaluation(100)
            ]

            features = base_features + feature_sets[j]

            X_train = df_train[features].to_numpy()
            y_train = df_train['responder_6'].to_numpy()
            weight_train = df_train['weight'].to_numpy()
            X_validate = df_validate[features].to_numpy()
            weight_validate = df_validate['weight'].to_numpy()

            del df_train

            lgb_eval_set = [(
                X_validate,
                df_validate['responder_6'].to_numpy()
            )]

            model = lgb.LGBMRegressor(n_estimators=round(50 / parameters['learning_rate']), objective='mse', **parameters)

            model.fit(
                X=X_train,
                y=y_train,
                sample_weight=weight_train,
                eval_metric=[r2_lgb],
                eval_set=lgb_eval_set,
                eval_sample_weight=[weight_validate],
                callbacks=lgb_callbacks
            )

            # get the feature importance (gain divided by the sum of gains) of the added model and save this in another list
            importance_scores = model.booster_.feature_importance(importance_type='gain')

            df_feature_importance = pl.DataFrame({
                'Feature': features,
                'Importance': importance_scores
            })

            df_feature_importance = df_feature_importance.with_columns(
                (pl.col('Importance') / pl.col('Importance').sum()).alias('Importance')
            )

            result_feature_set.append(model.best_score_['valid_0']['r2'])
            if j == 0:
                result_feature_importance_feature_set.append(0)
            else:
                result_feature_importance_feature_set.append(df_feature_importance.filter(pl.col('Feature').is_in(feature_sets[j])).select('Importance').to_series().sum())
            print(f'{datetime.now().strftime("%H:%M:%S")} . . . Model {i} fold {fold} feature set {j} done.')

        result_model.extend(result_feature_set)
        result_feature_importance_model.extend(result_feature_importance_feature_set)

    result.append(result_model)
    result_feature_importance.append(result_feature_importance_model)

col_names = [
    'val_1_base', 'val_2_base',
    'val_1_std', 'val_2_std',
    'val_1_weight', 'val_2_weight'
]
df_result = pl.DataFrame(result, col_names, orient='row')

df_result = df_result.with_columns(
    val_base=pl.mean_horizontal('val_1_base', 'val_2_base'),
    val_std=pl.mean_horizontal('val_1_std', 'val_2_std'),
    val_weight=pl.mean_horizontal('val_1_weight', 'val_2_weight')
)

# transpose to see ranking in data view
df_result_transpose = df_result.select(['val_base', 'val_std', 'val_weight']).transpose()


df_result_feature_importance = pl.DataFrame(result_feature_importance, col_names, orient='row')

df_result_feature_importance = df_result_feature_importance.with_columns(
    val_base=pl.mean_horizontal('val_1_base', 'val_2_base'),
    val_std=pl.mean_horizontal('val_1_std', 'val_2_std'),
    val_weight=pl.mean_horizontal('val_1_weight', 'val_2_weight')
)

# transpose to see ranking in data view
df_result_feature_importance_transpose = df_result_feature_importance.select(['val_base', 'val_std', 'val_weight']).transpose()

# save_as_pickle(result, DATA_PATH / "feature_addition_result.pkl")

