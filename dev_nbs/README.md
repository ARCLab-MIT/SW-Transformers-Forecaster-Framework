# Implementation Contents 👨‍💻

This notebook recopiles all the core implementation of the repository such as the model training, evaluation, execution and also the data preparation and hyperparameter tunning. For better understand the files in the repository they are named with the following rules:
```bash
{model indices}_[extra_info]_{nb_type}
```
## Indices Nomenclature ☀️🌍
- `solfsmy`: Refering to multivariate models over FSMY 10.7 solar indices.
- `geodst`: Refering to models only over Dst geomagnetic index.
- `geoap`: Refering to models only over Ap geomagnetic index.
- `geodstap`: Refering to multivariate models that combine Ap and Dst.
- `solf10`: There are only one file with this nomenclature, where the F10 historical data is prepared.

> [!NOTE]
> Some notebooks has some extra info as:
> * `tsr` where a trend, seasonal and residual decomposition is made.
> * `ensemble` where the model consist in an ensemble of several models trained using different loss functions.
> * `losses` where a losses study is taking place.

## Notebook Types 📓
* `data`: In these notebooks is where the data preparation is made.
* `train`: In these notebooks is where the models are being trained.
* `eval`: In these notebooks is where it can be evaluated the training performance against the test set.
* `preds`: In these notebooks some predictions are runned to check the model performance against updated data.
* `optuna_study`: Here is where an hyperparameters tunning study is made.

## Folders 📁
* [`config`](/dev_nbs/config/): Here are saved all the configuration files for each architecture, driver forecaster and data used in the implementation. These files recopile some static information that is being used during the different files, making it easier to change the setup without having to modify the code. 
* [`preprocessed_data`](/dev_nbs/preprocessed_data/): In this folder are saved all the data that can be reused in different notebooks, such as static data like thresholds or categories, or the preprocessed dataframes and the pipelines that have been applied to them.
* [`figures`](/dev_nbs/figures/): In this notebook are being save some of the figures that can be generated when using matplot.

> [!NOTE]
> During execution other folders will be generated, like:
> * `models` where the models are saved once training process has finished.
> * `tmp` using to share temporal data betweem notebooks.
> * `wandb` where all the wandb offline information is being saved.