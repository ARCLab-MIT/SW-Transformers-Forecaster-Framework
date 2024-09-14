# Implementation Contents 👨‍💻

This folder contains all the core implementations of the repository, including model training, evaluation, execution, data preparation, and hyperparameter tuning. To better understand the files in the repository, they follow the naming convention outlined below:

```bash
{model indices}_[extra_info]_{nb_type}
```
## Indices Nomenclature ☀️🌍
- `solfsmy`: Refers to multivariate models using FSMY 10.7 solar indices.
- `geodst`: Refers to models using only the Dst geomagnetic index.
- `geoap`: Refers to models using only the Ap geomagnetic index.
- `geodstap`: Refers to multivariate models that combine Ap and Dst.
- `solf10`: Refers to the file where the historical F10 data is prepared (only one file with this nomenclature).

> [!NOTE]
> Some notebooks include extra information such as:
> * `tsr`: Where trend, seasonal, and residual decomposition is applied.
> * `ensemble`: Where the model consists of an ensemble of several models trained with different loss functions.
> * `losses`: Where a loss function study is conducted.

## Notebook Types 📓
* `data`: Notebooks where data preparation is handled.
* `train`: Notebooks where models are trained.
* `eval`: Notebooks where training performance is evaluated against the test set.
* `preds`: Notebooks where predictions are generated to assess model performance on updated data.
* `optuna_study`: Notebooks for hyperparameter tuning studies.

## Folders 📁
* [`config`](/dev_nbs/config/): Stores configuration files for each architecture, forecaster, and dataset used in the implementation. These files collect static information used across the project, making it easier to modify the setup without altering the code.
* [`preprocessed_data`](/dev_nbs/preprocessed_data/): Contains data that can be reused across different notebooks, such as static data (e.g., thresholds or categories) and preprocessed dataframes and pipelines.
* [`figures`](/dev_nbs/figures/): Stores figures generated using Matplotlib.

> [!NOTE]
> During execution, additional folders will be generated, such as:
> * `models`: Stores models once the training process is completed.
> * `tmp`: Used to share temporary data between notebooks.
> * `wandb`: Stores all offline Weights and Biases (wandb) data.
