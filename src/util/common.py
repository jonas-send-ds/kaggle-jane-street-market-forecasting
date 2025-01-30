import random
import math
import numpy as np
import pickle

from typing import Any
from pathlib import Path


# Custom R2 metric for LightGBM
def r2_lgb(y_true, y_pred, sample_weight):
    y_pred_clipped = np.clip(y_pred, -5, 5)
    r2 = 1 - np.average((y_pred_clipped - y_true) ** 2, weights=sample_weight) / (np.average(y_true ** 2, weights=sample_weight) + 1e-38)  # add small number to avoid division by zero
    return 'r2', r2, True


def sample_parameters(space: dict[str, list[Any]]) -> dict[str, Any]:
    parameters: dict[str, Any] = dict()
    for key, value in space.items():
        if key == "num_leaves":
            parameters[key] = sample_leaves(parameters["max_depth"], value)
        else:
            parameters[key] = sample_parameter(value)
    return parameters


def sample_parameter(space: list[Any]) -> Any:
    return random.choice(space)


def sample_leaves(depth: int, space: list[int]) -> int:
    if depth == -1:
        leaves = random.choice(space)
    elif depth == 2:
        leaves = 3
    else:
        leaves = random.choice(get_leaf_space(depth, space))
    return leaves


def get_leaf_space(depth: int, space: list[int]) -> list[int]:
    potential_leave_space: list[int] = [2 ** x - 1 for x in list(range(math.ceil(math.log(depth, 2)), depth + 1))]
    return list(set(space) & set(potential_leave_space))


def save_as_pickle(x: Any, path: str | Path) -> None:
    open_file = open(path, "wb")
    pickle.dump(x, open_file)
    open_file.close()


def load_pickle(path: str | Path) -> Any:
    open_file = open(path, "rb")
    x = pickle.load(open_file)
    open_file.close()
    return x
