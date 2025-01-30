
import polars as pl

from src.util.constants import DATA_PATH

# loop over the train sets, concatenate them and save the result
df_train = pl.DataFrame({})

for i in range(10):
    df_train_loop = pl.read_parquet(f"{DATA_PATH}/raw/train.parquet/partition_id={i}/part-0.parquet")
    df_train = pl.concat([df_train, df_train_loop])

df_train.write_parquet(f"{DATA_PATH}/df_all.parquet")
