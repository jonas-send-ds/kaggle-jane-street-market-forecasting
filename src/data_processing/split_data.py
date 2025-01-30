
import polars as pl

from src.util.constants import DATA_PATH


DATES_TO_PURGE = 5 * 4  # 4 weeks

df_all = pl.read_parquet(DATA_PATH / 'df_all_with_features.parquet')

no_observation = df_all.shape[0]
number_of_observation = df_all.shape[0]
# cut into 20 blocks and choose representative data (see performance_over_time.py)
cutoff_dates = [df_all['date_id'][round(number_of_observation*x/20)] for x in range(20)]  + [df_all['date_id'].max() + 1]

# train & validate 1 (blocks 8 and 17 (zero-indexed))
df_all.filter(
    (pl.col('date_id') < cutoff_dates[8] - DATES_TO_PURGE) |
    ((pl.col('date_id') >= cutoff_dates[9] + DATES_TO_PURGE) & (pl.col('date_id') < cutoff_dates[17] - DATES_TO_PURGE)) |
    (pl.col('date_id') >= cutoff_dates[18] + DATES_TO_PURGE)
).write_parquet(DATA_PATH / 'df_train_1_with_features.parquet')
df_all.filter(
    ((pl.col('date_id') >= cutoff_dates[8]) & (pl.col('date_id') < cutoff_dates[9])) |
    ((pl.col('date_id') >= cutoff_dates[17]) & (pl.col('date_id') < cutoff_dates[18]))
).write_parquet(DATA_PATH / 'df_validate_1_with_features.parquet')

# train & validate 2
df_all.filter(
    (pl.col('date_id') < cutoff_dates[9] - DATES_TO_PURGE) |
    ((pl.col('date_id') >= cutoff_dates[10] + DATES_TO_PURGE) & (pl.col('date_id') < cutoff_dates[18] - DATES_TO_PURGE)) |
    (pl.col('date_id') >= cutoff_dates[19] + DATES_TO_PURGE)
).write_parquet(DATA_PATH / 'df_train_2_with_features.parquet')
df_all.filter(
    ((pl.col('date_id') >= cutoff_dates[9]) & (pl.col('date_id') < cutoff_dates[10])) |
    ((pl.col('date_id') >= cutoff_dates[18]) & (pl.col('date_id') < cutoff_dates[19]))
).write_parquet(DATA_PATH / 'df_validate_2_with_features.parquet')

# train & validate 3 (test set)
df_all.filter(
    (pl.col('date_id') < cutoff_dates[10] - DATES_TO_PURGE) |
    ((pl.col('date_id') >= cutoff_dates[11] + DATES_TO_PURGE) & (pl.col('date_id') < cutoff_dates[19] - DATES_TO_PURGE))
).write_parquet(DATA_PATH / 'df_train_3_with_features.parquet')
df_all.filter(
    ((pl.col('date_id') >= cutoff_dates[10]) & (pl.col('date_id') < cutoff_dates[11])) |
    (pl.col('date_id') >= cutoff_dates[19])
).write_parquet(DATA_PATH / 'df_validate_3_with_features.parquet')
