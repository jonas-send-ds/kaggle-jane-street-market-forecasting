
import polars as pl
import random

from datetime import datetime
from src.util.constants import NUMBER_OF_SHADOW_FEATURES, DATA_PATH
from src.util.common import save_as_pickle


SHADOW_SUFFIX: str = "_shadow"

feature_names = ([f"feature_{i:02d}" for i in range(79)] +
                 ['responder_6_mean_lag_1', 'responder_6_std_lag_1'] +
                 [f'responder_{i}_mean_lag_1' for i in range(9) if i != 6])
random_features = random.sample(feature_names, NUMBER_OF_SHADOW_FEATURES)
feature_names_with_shadow_features = feature_names + [name + SHADOW_SUFFIX for name in random_features]

save_as_pickle(feature_names_with_shadow_features, str(DATA_PATH / "feature_names.pkl"))


def add_and_save_shadow_features(name: str, _df: pl.DataFrame, _feature_names: list[str]) -> None:
    # randomise each feature column and rename
    shadow_df = _df.select([pl.col(col_name).shuffle().alias(col_name + SHADOW_SUFFIX) for col_name in _feature_names])
    _df_with_shadow = pl.concat([_df, shadow_df], how="horizontal")

    _df_with_shadow.write_parquet(DATA_PATH / f"df_{name}_with_shadow_features.parquet")

    print(f'{datetime.now().strftime("%H:%M:%S")} . . . Shadow features saved for {name}.')


for fold in range(1, 3):
    df_train: pl.DataFrame = pl.read_parquet(DATA_PATH / f"df_train_{fold}_with_lags.parquet")
    df_validate: pl.DataFrame = pl.read_parquet(DATA_PATH / f"df_validate_{fold}_with_lags.parquet")

    add_and_save_shadow_features(f"train_{fold}_with_lags", df_train, random_features)
    add_and_save_shadow_features(f"validate_{fold}_with_lags", df_validate, random_features)
