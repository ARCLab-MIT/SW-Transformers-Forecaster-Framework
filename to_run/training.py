# -*- coding: utf-8 -*-
"""training.ipynb

Automatically generated.

Original file is located at:
    dev_nbs/training.ipynb
"""

import sklearn
from tsai.basics import *
from swdf.utils import *
my_setup(sklearn)
from matplotlib import dates as mdates
import wandb
from fastai.callback.wandb import WandbCallback
from fastai.callback.progress import ShowGraphCallback

config = AttrDict(
    add_time_channels = False, # Add time channels to the data (year, day of year)
    arch_config = AttrDict(
        n_layers=3,  # number of encoder layers
        n_heads=4,  # number of heads
        d_model=16,  # dimension of model
        d_ff=128,  # dimension of fully connected network
        attn_dropout=0.0, # dropout applied to the attention weights
        dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
        patch_len=9,  # length of the patch applied to the time series to create patches. 
        stride=1,  # stride used when creating patches
        padding_patch=True,  # padding_patch
    ),
    bs = 16, # Batch size
    data_columns_fcst = ['F10', 'S10', 'M10', 'Y10'], # Columns to use for forecasting
    data_url = 'https://sol.spacenvironment.net/jb2008/indices/SOLFSMY.TXT',
    data_path = '../data/SOLFSMY.TXT',
    init_weights = True, # Kaiming init weights
    n_epoch = 5, # Number of epochs to train for
    lookback = 36, # six times the horizon, as in Stevenson et al. (2021)
    horizon = 6, # same as paper by Licata et al. (2020)
    use_wandb = True, # To use it, the environment variable WANDB_API_KEY must be set
    wandb_project = 'swdf', # Name of wandb project
)

run = wandb.init(project=config.wandb_project, 
                 config=config, 
                 anonymous='never') if config.use_wandb else None
config = run.config if config.use_wandb else config

fname = config.data_path if config.data_url is None else download_data(config.data_url,
                                                                       fname=config.data_path)
fname

# Read the text file into a pandas DataFrame, ignoring the lines starting with '#'
# Column names: YYYY DDD   JulianDay  F10   F81c  S10   S81c  M10   M81c  Y10   Y81c  Ssrc
df_raw = pd.read_csv(fname, delim_whitespace=True, comment='#', header=None, 
                 names=['Year', 'DDD', 'JulianDay', 'F10', 'F81c', 'S10', 'S81c', 
                        'M10', 'M81c', 'Y10', 'Y81c', 'Ssrc'])
df_raw.head()

# Check if there are any missing values
df_raw.isna().sum()

# Distinct value of the column Ssrc
df_raw.Ssrc.unique()

# Separate the Ssrc columns into four colums, one for each character of the string,
# The names of the new columns will be SsrcF10, SsrcS10, SsrcM10, and SsrcY10,
# Cast the new columns into categories. Use a loop
for i, c in enumerate('F10 S10 M10 Y10'.split()):
    df_raw[f'Ssrc_{c}'] = df_raw['Ssrc'].str[i].astype('category')
df_raw[['Ssrc_F10', 'Ssrc_S10', 'Ssrc_M10', 'Ssrc_Y10']].head()

# See the categories of the column Ssrc_S10
df_raw.Ssrc_S10.cat.categories

# Convert the JulianDay column to a datetime column, and set it as index
df_raw['Date'] = pd.to_datetime(df_raw['JulianDay'], unit='D', origin='julian')
df_raw['Date'].head()

# Get the number of values equlas to zero in S10
print((df_raw.S10 == 0).sum())
# convert them to NA
df_raw.loc[df_raw.S10 == 0, 'S10'] = np.nan
print((df_raw.S10 == 0).sum())

datetime_col = 'Date'
freq = '1D'
data_columns_fcst = config.data_columns_fcst
data_columns_time = ['Year', 'DDD']
data_columns = data_columns_fcst + data_columns_time if config.add_time_channels else data_columns_fcst
imputation_method = 'ffill'

# sklearn's preprocessing pipeline
preproc_pipe = sklearn.pipeline.Pipeline([
    ('shrinker', TSShrinkDataFrame()), # shrik dataframe memory usage and set the right dtypes
    ('drop_duplicates', TSDropDuplicates(datetime_col='Date')), # drop duplicates
    ('fill_missing', TSFillMissing(columns=data_columns, method=imputation_method, value=None)), # fill missing data (1st ffill. 2nd value=0)
], verbose=True)

df = preproc_pipe.fit_transform(df_raw)
df

# In the paper by Licata et al. (2020) (https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2020SW002496),
# authors use a period from October 2012 through the end of 2018 for the benchmarking.
# Therefore, we will set the test set as the same period for our analysis, 
# using the column Date as the timestamp, from October 2012 to the end of 2018. 
# Everything before the test set will be used for training, and everything after the test set
# will be used for validation
test_start_datetime = '2012-10-01'
test_end_datetime = '2018-12-31'
valid_start_datetime = '2018-01-01'

# Splits: Since the validation period is after the test period in this use case, we cannot
# use the default `get_forecasting_splits` from tsai. Instead, we will do manually
# the validation splits, and use the funcion only for the test splits

#val_idxs = L(df[df.Date >= valid_start_datetime].index.tolist())
splits_ = get_forecasting_splits(df[df.Date < valid_start_datetime], 
                             fcst_history=config.lookback, 
                             fcst_horizon=config.horizon, 
                             use_index=False, 
                             test_cutoff_datetime=test_start_datetime, 
                             show_plot=False, 
                             datetime_col='Date')
foo = df[df.Date >= valid_start_datetime]
bar = get_forecasting_splits(foo, config.lookback, config.horizon, valid_size=0.0, 
                             test_size=0.0, show_plot=False)
val_idxs = L(foo.index[bar[0]].tolist())

splits = (splits_[0], val_idxs, splits_[1])
splits

# Now that we have defined the splits for this particular experiment, we'll scale
# the data
train_split = splits[0]
exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=data_columns)),
], verbose=True)
save_object(exp_pipe, 'tmp/exp_pipe.pkl')
exp_pipe = load_object('tmp/exp_pipe.pkl')
# TODO: I don't know why but if I don't copy the dataframe df it gets modified
df_scaled = exp_pipe.fit_transform(df.copy(), scaler__idxs = train_split)
#df_scaled.set_index(datetime_col, inplace=True)
df_scaled.head()

# We'll approach the time series forecasting task as a supervised learning problem. 
# Remember that tsai requires that both inputs and outputs have the following shape:
# (samples, features, steps)

# To get those inputs and outputs we're going to use a function called 
# `prepare_forecasting_data`` that applies a sliding window along the dataframe
X, y = prepare_forecasting_data(df, fcst_history=config.lookback, fcst_horizon=config.horizon, 
                                x_vars=data_columns, y_vars=data_columns)
X.shape, y.shape

cbs = L(WandbCallback(log_preds=False)) if config.use_wandb else L()
learn = TSForecaster(X, y, splits=splits, batch_size=config.bs, 
                     pipelines=[preproc_pipe, exp_pipe], arch="PatchTST", 
                     arch_config=dict(config.arch_config), 
                    cbs= cbs + ShowGraphCallback())
#lr_max = learn.lr_find().valley
lr_max = 1e-3
print(f"#params: {sum(p.numel() for p in learn.model.parameters())}")

cbs = L(WandbCallback(log_preds=False)) if config.use_wandb else L()
learn = TSForecaster(X, y, splits=splits, batch_size=config.bs, 
                     pipelines=[preproc_pipe, exp_pipe], arch="PatchTST", 
                     arch_config=dict(config.arch_config), 
                     init=config.init_weights,
                    cbs= cbs + ShowGraphCallback())
#lr_max = learn.lr_find().valley
lr_max = 1e-3
print(f"#params: {sum(p.numel() for p in learn.model.parameters())}")
learn.fit_one_cycle(n_epoch=config.n_epoch, lr_max=1e-3)
print(learn.validate())

learn.dls.loaders += [learn.dls.valid.new_dl(X[splits[2]], y[splits[2]])] # Add test datalaoder
learn.save_all(path='tmp', verbose=True) 
if run is not None:
    run.log_artifact('tmp/learner.pkl')