
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import src.util.common as common

from lightgbm import early_stopping, log_evaluation
from datetime import datetime

from src.util.constants import DATA_PATH
from src.util.common import r2_lgb


NUMBER_OF_SLICES = 10
NUMBER_OF_MODELS = 10
DATES_TO_PURGE = 5 * 4  # 4 weeks


df_all = pl.read_parquet(DATA_PATH / 'df_all.parquet')

feature_names = [f"feature_{i:02d}" for i in range(79)]

number_of_observation = df_all.shape[0]
dates = df_all['date_id'].unique().to_list()
start_dates = [df_all['date_id'][round(number_of_observation*x/NUMBER_OF_SLICES)] for x in range(NUMBER_OF_SLICES)]  + [df_all['date_id'].max() + 1]

fixed_parameters = {
    "learning_rate": .1,
    "n_jobs": 12,  # current number of cores for Mac
    "subsample_freq": 1,
    "verbose": -1,
    "metric": "None"
}

hyperparameter_space = {"learning_rate": [.05, .1],
                        'max_bin': [2 ** x - 1 for x in list(range(5, 11))],
                        "max_depth": list(range(4, 12)),
                        "num_leaves": [2 ** x - 1 for x in list(range(6, 10))],
                        "bagging_fraction": [.4, .5, .6, .7, .8, .9],
                        "feature_fraction": [.3, .4, .5, .6, .7, .8],
                        "min_data": [100, 200, 500, 1000, 5000, 10000],
                        "min_sum_hessian_in_leaf": [.00001, .0001, .001, .005, .01, .1, 1, 10, 100, 1000],
                        "lambda_l1": [0, .00001, .0001, .001, .005, .01, .1, 1, 10],
                        "lambda_l2": [.1, 1, 10, 100, 1000, 10000, 100000]}

results = []
for i in range(NUMBER_OF_MODELS):
    sampled_parameters = common.sample_parameters(hyperparameter_space)
    parameters = {**fixed_parameters, **sampled_parameters}

    result = []
    for j in range(NUMBER_OF_SLICES):
        train_dates = [x for x in dates if x < (start_dates[j] - DATES_TO_PURGE) or x > (start_dates[j+1] + DATES_TO_PURGE)]
        validate_dates = [x for x in dates if start_dates[j] <= x < start_dates[j+1]]

        df_train = df_all.filter(pl.col('date_id').is_in(train_dates))
        df_validate = df_all.filter(pl.col('date_id').is_in(validate_dates))

        lgb_callbacks = [
            early_stopping(stopping_rounds=10, first_metric_only=True, verbose=False),
            log_evaluation(100)
        ]

        X_train = df_train[feature_names].to_numpy()
        y_train = df_train['responder_6'].to_numpy()
        weight_train = df_train['weight'].to_numpy()
        X_validate = df_validate[feature_names].to_numpy()
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

        df_validate.with_columns(
            prediction=model.predict(X_validate)
        )

        result_loop = []
        for date in validate_dates:
            df_validate_date = df_validate.filter(pl.col('date_id') == date)
            r2 = model.score(df_validate_date[feature_names].to_numpy(), df_validate_date['responder_6'].to_numpy(), sample_weight=df_validate_date['weight'].to_numpy())
            result_loop.append(r2)

        result.extend(result_loop)

        print(f'{datetime.now().strftime("%H:%M:%S")} . . . Slice {j} done.')

    results.append(result)

    print(f'{datetime.now().strftime("%H:%M:%S")} . . . Model {i} done.')

df_results = pl.DataFrame(
    results,
    [f'model_{i}' for i in range(NUMBER_OF_MODELS)],
    orient='col'
)

df_results.write_parquet(DATA_PATH / 'df_results_performance_over_time.parquet')

df_results.corr()
# the performance of models is highly correlated (> .93)

# look at average performance of models over time
df_results_copy = df_results
for i in range(NUMBER_OF_MODELS):
    df_results_copy = df_results_copy.with_columns(
        pl.col(f"model_{i}").rolling_mean(window_size=30).alias(f"mean_{i}")
    )
    plt.plot(df_results_copy[f'mean_{i}'])

plt.show()

# see how models are correlated across different periods


# Cut the set into 20 similar sized pieces.
# I want to select three consecutive blocks for my three validation sets in the two lowly correlated bigger blocks.

start_dates_20 = [df_all['date_id'][round(number_of_observation*x/20)] for x in range(20)]  + [df_all['date_id'].max() + 1]
group_lengths = [start_dates_20[i+1]-start_dates_20[i] for i in range(len(start_dates_20)-1)]
group_labels = [i for i, length in enumerate(group_lengths) for _ in range(length)]

# add index column for NUMBER_OF_GROUPS groups
df_results = df_results.with_columns(
    index=pl.Series(group_labels)
)
# group by index and get column-wise mean
df_performance_by_group = df_results.group_by('index', maintain_order=True).agg(pl.all().mean())[:, 1:NUMBER_OF_MODELS+1].transpose()

df_corr = df_performance_by_group.corr()

plt.figure(figsize=(20, 12))

# show the values in the heatmap, rounded to two decimals
sns.heatmap(df_corr.to_numpy(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
