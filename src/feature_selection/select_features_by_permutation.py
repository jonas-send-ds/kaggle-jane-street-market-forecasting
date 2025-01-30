
import polars as pl
import lightgbm as lgb
import lleaves
import src.util.common as common

from lightgbm import early_stopping, log_evaluation
from datetime import datetime
from src.util.common import r2_lgb, save_as_pickle, load_pickle
from src.util.constants import DATA_PATH, NUMBER_OF_SHADOW_FEATURES


df_train_list = [
    pl.read_parquet(DATA_PATH / 'df_train_1_with_lags_with_shadow_features.parquet'),
    pl.read_parquet(DATA_PATH / 'df_train_2_with_lags_with_shadow_features.parquet')
]
df_validate_list = [
    pl.read_parquet(DATA_PATH / 'df_validate_1_with_lags_with_shadow_features.parquet'),
    pl.read_parquet(DATA_PATH / 'df_validate_2_with_lags_with_shadow_features.parquet')
]

feature_names = load_pickle(str(DATA_PATH / "feature_names.pkl"))
fixed_variables = ['responder_6', 'weight']

fixed_parameters = {
    "learning_rate": .1,
    "n_jobs": 12,  # current number of cores for Mac
    "subsample_freq": 1,
    "verbose": -1,
    "metric": "None"
}

hyperparameter_space = {"learning_rate": [.05, .1],
                        'max_bin': [2 ** x - 1 for x in list(range(5, 10))],
                        "max_depth": list(range(8, 12)),
                        "num_leaves": [2 ** x - 1 for x in list(range(7, 10))],
                        "bagging_fraction": [.1,.2, .3, .4, .5, .6, .7],
                        "feature_fraction": [.2, .3, .4, .5, .6, .7, .8],
                        "min_data": [50, 100, 200, 500, 1000, 5000],
                        "min_sum_hessian_in_leaf": [.0001, .001, .005, .01, .1, 1, 10, 100, 1000],
                        "lambda_l1": [0, .00001, .0001, .001, .005, .01, .1, 1, 10],
                        "lambda_l2": [10, 100, 1000, 10000, 100000]}

# load dataset if it exists
try:
    df_features = pl.read_parquet(DATA_PATH / "df_feature_selection_result.parquet")
    i = int((df_features.shape[1] - 2) / 2)
    print(f"Loaded existing feature selection data with {i} iterations.")
except FileNotFoundError:
    df_features = pl.DataFrame({"feature": feature_names, "active": True})
    df_features = df_features.with_columns(
        (pl.col("feature").str.ends_with("_shadow")).alias("shadow"),
        mean=0
    )
    i = 0
    print("Initialised new feature selection data.")

def update_active_features(_df_features: pl.DataFrame) -> pl.DataFrame:
    columns_df_feature = ['feature', 'shadow', 'active', 'mean']

    if _df_features.shape[1] <= len(columns_df_feature):
        return _df_features

    _df_features = _df_features.with_columns(
        pl.concat_list(
            [pl.col(col) for col in _df_features.columns if col not in columns_df_feature]
        ).list.mean().alias('mean')
    )

    max_shadow_mean = _df_features.filter(pl.col('shadow'))['mean'].max()
    _df_features = _df_features.with_columns(
        active=pl.when(pl.col('shadow'))
        .then(True)
        .otherwise(pl.col('mean') > max_shadow_mean)
    )

    return _df_features


def get_active_features(_df_features: pl.DataFrame) -> list[str]:
    return _df_features.filter(pl.col('active')).select('feature').to_series().to_list()


active_features_after = get_active_features(df_features)

# initialise above threshold
nr_dropped_features = len(get_active_features(df_features))
current_iteration = 0
iterations = 20  # this will be decreasing

while nr_dropped_features >= len(feature_names) * 0.025:  # go on if we have dropped more than 2.5% of features in the last round
    current_iteration += 1

    sampled_parameters = common.sample_parameters(hyperparameter_space)
    parameters = {**fixed_parameters, **sampled_parameters}

    active_features_before = get_active_features(df_features)

    for fold in range(2):
        df_train = df_train_list[fold]
        df_validate = df_validate_list[fold]

        lgb_callbacks = [
            early_stopping(stopping_rounds=10, first_metric_only=True, verbose=False),
            log_evaluation(100)
        ]

        X_train = df_train[active_features_before].to_numpy()
        y_train = df_train['responder_6'].to_numpy()
        weight_train = df_train['weight'].to_numpy()
        X_validate = df_validate[active_features_before].to_numpy()
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

        result = []

        # compile model to lleaves
        model.booster_.save_model(DATA_PATH / f'models/tmp/model_{i}.txt')
        llvm_model = lleaves.Model(model_file=DATA_PATH / DATA_PATH / f'models/tmp/model_{i}.txt')
        llvm_model.compile(cache=DATA_PATH / f'models/tmp/model_{i}.o')

        prediction = pl.Series(llvm_model.predict(df_validate[active_features_before].to_numpy()))
        model_score = r2_lgb(df_validate['responder_6'].to_numpy(), prediction, df_validate['weight'].to_numpy())[1]

        for k in range(len(feature_names)):
            feature_name = feature_names[k]

            if feature_name in active_features_before:
                feature_copy = df_validate[feature_name]

                df_validate = df_validate.with_columns(pl.col(feature_name).shuffle().alias(feature_name))

                prediction = pl.Series(llvm_model.predict(df_validate[active_features_before].to_numpy()))
                # noinspection PyTypeChecker
                permutation_score = r2_lgb(df_validate['responder_6'].to_numpy(), prediction, df_validate['weight'].to_numpy())[1]

                result.append(model_score - permutation_score)

                df_validate = df_validate.with_columns(feature_copy)
            else:
                result.append(-1.0)

        df_features = df_features.with_columns(pl.Series(result).alias(f'model_{i}_fold_{fold}'))

        print(f'{datetime.now().strftime("%H:%M:%S")} . . . Model {i} fold {fold} done.')

    # drop 'bad' features
    if current_iteration >= iterations:
        nr_active_features_before = len(active_features_before)

        df_features = update_active_features(df_features)
        active_features_after = get_active_features(df_features)

        for fold in range(2):
            df_train_list[fold] = df_train_list[fold][fixed_variables + active_features_after]
            df_validate_list[fold] = df_validate_list[fold][fixed_variables + active_features_after]

        nr_dropped_features = nr_active_features_before - len(active_features_after)

        current_iteration = 0
        iterations = round(iterations * 0.75)

    df_features.write_parquet(str(DATA_PATH) + "/df_feature_selection_result.parquet")
    i += 1

df_features = update_active_features(df_features)
feature_selection_list = list(get_active_features(df_features)[:-NUMBER_OF_SHADOW_FEATURES])

save_as_pickle(feature_selection_list, str(DATA_PATH) + "/feature_selection_list.pkl")

feature_selection_list = feature_selection_list + fixed_variables

# All raw features remain active

# del df_train_list, df_validate_list  # make some room
#
# for name in ["train_1", "validate_1", "train_2", "validate_2", "train_3", "validate_3"]:
#     df = pl.read_parquet(str(DATA_PATH) + "/df_" + name + "_with_lags.parquet")
#     df = df[feature_selection_list]
#     df.write_parquet(str(DATA_PATH) + "/df_" + name + "_with_lags_selected_features.parquet")
