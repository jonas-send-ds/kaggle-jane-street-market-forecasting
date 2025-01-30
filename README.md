# kaggle-jane-street-market-forecasting
Requires the data downloaded from https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data in the folder `data`.

### Python / Conda Environment
The packages, including versions, are documented in the `environment.yml` file.
The `environment.yml` file is created/updated using:
```console
conda env export --no-builds | grep -v '^prefix:' > environment.yml
```
(The grep removes the prefix, which is only useful on a specific machine.)

The environment can be replicated from this yaml file using:
```console
conda env create -f environment.yml
```

To update an existing environment from a yaml file (e.g. after new packages have been installed by another user), use:
```console
conda env update -f environment.yml --prune
```

Or, if that doesn't work, delete the environment first:
```console
conda deactivate
conda remove -n kaggle-jane-street --all 
```
and then re-create it.

