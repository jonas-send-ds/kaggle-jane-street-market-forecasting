
from pathlib import Path


def find_project_root(current_path, markers=('src', 'data')):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if all((parent / marker).exists() for marker in markers):
            return parent
    raise FileNotFoundError("Project root not found.")


DATA_PATH: Path = find_project_root(Path.cwd()) / 'data'
NUMBER_OF_SHADOW_FEATURES: int = 39  # half the number of original features
