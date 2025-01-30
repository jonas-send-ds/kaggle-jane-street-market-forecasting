
import polars as pl

from src.util.constants import DATA_PATH

df_all = pl.read_parquet(DATA_PATH / 'df_all.parquet')

# Mean, min, max, and std over feature columns
feature_columns = [f'feature_{i:02}' for i in range(79) if i not in [9, 10, 11]]  # exclude the integer columns
df_all = df_all.with_columns(
    feature_mean=pl.mean_horizontal(feature_columns),
    feature_min=pl.min_horizontal(feature_columns),
    feature_max=pl.max_horizontal(feature_columns),
    feature_std=pl.concat_list(feature_columns).list.std()
)

# Lag of one-day mean of each responder (by symbol)
for i in range(9):
    df_all = df_all.with_columns(
        pl.col(f'responder_{i}').mean().over(['date_id', 'symbol_id']).alias(f'responder_{i}_mean')
    )
    df_all = df_all.with_columns(
        pl.col(f'responder_{i}_mean').shift(1).over(['time_id', 'symbol_id']).alias(f'responder_{i}_mean_lag_1')
    ).drop(
        f'responder_{i}_mean'
    )

# normalise time
df_all = df_all.with_columns(
    pl.col('time_id').max().over(['date_id']).alias('time_id_max')
)
df_all = df_all.with_columns(
    time=pl.col('time_id') / pl.col('time_id_max')
).drop(
    'time_id_max'
)

df_all.write_parquet(DATA_PATH / 'df_all_with_features.parquet')
