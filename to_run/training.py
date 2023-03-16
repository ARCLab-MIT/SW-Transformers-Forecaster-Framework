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
    data_url = 'https://sol.spacenvironment.net/jb2008/indices/SOLFSMY.TXT',
    data_path = '../data/SOLFSMY.TXT',
    n_epoch = 10, # Number of epochs to train for
    lookback = 36, # six times the horizon, as in Stevenson et al. (2021)
    horizon = 6, # same as paper by Licata et al. (2020)
    use_wandb = False, # To use it, the environment variable WANDB_API_KEY must be set
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

# Convert the JulianDay column to a datetime column, and set it as index
df_raw['Date'] = pd.to_datetime(df_raw['JulianDay'], unit='D', origin='julian')
df_raw['Date'].head()
df_raw.set_index('Date', inplace=True)

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

# Get the number of values equlas to zero in S10
print((df_raw.S10 == 0).sum())
# convert them to NA
df_raw.loc[df_raw.S10 == 0, 'S10'] = np.nan
print((df_raw.S10 == 0).sum())

datetime_col = 'Date'
freq = '1D'
data_columns = 'F10 S10 M10 Y10'.split()
imputation_method = 'ffill'

# sklearn's preprocessing pipeline
preproc_pipe = sklearn.pipeline.Pipeline([
    ('shrinker', TSShrinkDataFrame()), # shrik dataframe memory usage and set the right dtypes
    ('drop_duplicates', TSDropDuplicates(use_index=True)), # drop duplicates
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

#val_idxs = L(df.reset_index()[df.index >= valid_start_datetime].index.tolist())
splits_ = get_forecasting_splits(df[df.index < valid_start_datetime], 
                             fcst_history=config.lookback, 
                             fcst_horizon=config.horizon, 
                             use_index=True, 
                             test_cutoff_datetime=test_start_datetime, 
                             show_plot=False)
foo = df.reset_index()[df.index >= valid_start_datetime]
bar = get_forecasting_splits(foo, config.lookback, config.horizon, valid_size=0.0, 
                             test_size=0.0, show_plot=False)
val_idxs = L(foo.index[bar[0]].tolist())

splits = (splits_[0], val_idxs, splits_[1])
splits

# Now that we have defined the splits for this particular experiment, we'll scaled
# the data
train_split = splits[0]
exp_pipe = sklearn.pipeline.Pipeline([
    ('scaler', TSStandardScaler(columns=data_columns)),
], verbose=True)
save_object(exp_pipe, 'tmp/exp_pipe.pkl')
exp_pipe = load_object('tmp/exp_pipe.pkl')

df_scaled = exp_pipe.fit_transform(df.reset_index(), scaler__idxs = train_split)
df_scaled.set_index(datetime_col, inplace=True)
df_scaled.head()

# We'll approach the time series forecasting task as a supervised learning problem. 
# Remember that tsai requires that both inputs and outputs have the following shape:
# (samples, features, steps)

# To get those inputs and outputs we're going to use a function called 
# `prepare_forecasting_data`` that applies a sliding window along the dataframe:
x_vars = data_columns
y_vars = data_columns
X, y = prepare_forecasting_data(df, fcst_history=config.lookback, fcst_horizon=config.horizon, 
                                x_vars=x_vars, y_vars=y_vars)
X.shape, y.shape

cbs = L(WandbCallback(log_preds=False)) if config.use_wandb else L()
learn = TSForecaster(X, y, splits=splits, batch_size=config.bs, path="models", 
                     pipelines=[preproc_pipe, exp_pipe], arch="PatchTST", 
                     arch_config=dict(config.arch_config), metrics=[mse, mae], 
                     cbs= cbs + ShowGraphCallback())
lr_max = learn.lr_find().valley
print(f"#params: {sum(p.numel() for p in learn.model.parameters())}")

learn.fit_one_cycle(n_epoch=config.n_epoch, lr_max=lr_max)
learn.export('patchTST.pt')

from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_test = y[splits[2]]
learn = load_learner('models/patchTST.pt')
y_test_preds, *_ = learn.get_X_preds(X[splits[2]])
y_test_preds = to_np(y_test_preds)
print(f"y_test_preds.shape: {y_test_preds.shape}")     

#| export

# The forecast error is defined as the difference  between the forecasted value 
# and the actual value. It is also dependant on the time, i.e., on the horizon 
# of the forecast (h).
# NOTE: This function computes the error for just one variable, not for all of them
def forecast_error(y_true, y_pred, h):
    # y_true: actual values (samples x steps)
    # y_pred: predicted values (samples x steps)
    # h: horizon of the forecast (1..horizon)
    return (y_pred[:, h-1] - y_true[:, h-1])

# forecast error normalized by the actual value
def percent_forecast_error(y_true, y_pred, h):
    # y_true: actual values (samples x steps)
    # y_pred: predicted values (samples x steps)
    # h: horizon of the forecast (1..horizon)
    return 100*(forecast_error(y_true, y_pred, h) / y_true[:, h-1])

# Test
print(y_test.shape, 
      forecast_error(y_test[:,0,:], y_test_preds[:,0,:], 6).shape,
      percent_forecast_error(y_test[:,0,:], y_test_preds[:,0,:], 6).shape)

#|export

def get_idxs_per_solar_activity_level(data, thresholds):
    # function that splits the data of a variable into 4 different sets, 
    # one for each solar activity level. The data comes as a numpy array with 
    # shape (samples, steps), and the split is done along the samples axis. 
    # The decision is made based on the first column of each sample. The function 
    # returns a list of 4 numpy arrays, one for each solar activity level. 
    # But it does not return the values, it returns the indices of the
    # samples that belong to each solar activity level.
    idxs_per_solar_activity_level = []
    for i in range(len(thresholds) + 1):
        if i == 0:
            idxs = np.where(data[:, 0] <= thresholds[i])[0]
        elif i == len(thresholds):
            idxs = np.where(data[:, 0] > thresholds[i-1])[0]
        else:
            idxs = np.where((data[:, 0] > thresholds[i-1]) & (data[:, 0] <= thresholds[i]))[0]
        idxs_per_solar_activity_level.append(idxs)
    return idxs_per_solar_activity_level

# Call the function for each variable, using y_test
thresholds = {'F10': [75, 150, 190], 
              'S10': [65, 150, 215], 
              'M10': [72, 144, 167], 
              'Y10': [81, 148, 165]}

y_test_split_idxs = {}
for i, var in enumerate(data_columns):
    y_test_split_idxs[var] = get_idxs_per_solar_activity_level(y_test[:, i, :], 
                                                               thresholds[var])
# Check the shape of each variable
for var in data_columns:
    print(f"{var}: {[y_test_split_idxs[var][i].shape for i in range(4)]}")

# Now split y_test and y_test_preds with the idxs we got, using numpy.take
y_test_split = {}
y_test_preds_split = {}
for var in data_columns:
    y_test_split[var] = [y_test[:, i, :].take(y_test_split_idxs[var][i], axis=0) for i in range(4)]
    y_test_preds_split[var] = [y_test_preds[:, i, :].take(y_test_split_idxs[var][i], axis=0) for i in range(4)]

# Check the shape of each variable in y_test_split
for var in data_columns:
    print(f"y_test-{var}: {[y_test_split[var][i].shape for i in range(4)]}")
    print(f"y_test_preds-{var}: {[y_test_preds_split[var][i].shape for i in range(4)]}")

# Create a table (dataframe) with the mean forecast error for each variable (F10, S10, M10, Y10), 
# each solar activity level and each horizon (1..horizon)
data = []
sals = ['low', 'moderate', 'elevated', 'high']
for var_idx, var in enumerate(data_columns):
    for sal_idx,sal in enumerate(sals):
        for h in range(1, config.horizon+1):
            fe_sfu = forecast_error(y_test_split[var][sal_idx], 
                                y_test_preds_split[var][sal_idx], h)
            fe_percent = percent_forecast_error(y_test_split[var][sal_idx],
                                            y_test_preds_split[var][sal_idx], h)
            n_samples = y_test_split[var][sal_idx].shape[0]
            data.append([var, sal, h, np.mean(fe_sfu), np.std(fe_sfu), 
                        np.mean(fe_percent), np.std(fe_percent), n_samples])
df_results = pd.DataFrame(data, columns=['variable', 'condition', 'horizon', 
                                         'µ_sfu', 'σ_sfu', 'µ_percent', 'σ_percent', 'n_samples'])
df_results.head(10)

# Divide the table into one dataframe for each variable, and print it in a way
# that has the same format as the table in the paper, that is:
# Columns: | Condition | Statistics | 1 Day | 2 Days | 3 Days | ... | {{horizon}} Days,
# where condition is the variable and the solar activity level, and statistics 
# is the mean (column mean_fe) and the standard deviation (std_fe) of the forecast error.
for var in data_columns:
    df_var = df_results[df_results['variable'] == var].drop(columns=['variable', 
                                                                     'µ_percent', 
                                                                     'σ_percent'])
    df_var = df_var.melt(id_vars=['condition', 'horizon'], 
                         value_vars=['µ_sfu', 'σ_sfu'], 
                         var_name='Statistic')
    # Sort the values of the column condition so that the order is 
    # low, moderate, elevated, high
    df_var['condition'] = pd.Categorical(df_var['condition'], 
                                      categories=['low', 'moderate', 'elevated', 'high'], 
                                      ordered=True)
    df_var = df_var.pivot_table(index=['condition', 'Statistic'], 
                          columns='horizon', 
                          values='value')
    # pretty print, and separate with a blank line
    print(df_var.to_string(), '\n')

dtms = prepare_forecasting_data(df.reset_index(), fcst_history=config.lookback, 
                                fcst_horizon=config.horizon, x_vars='Date', y_vars='Date')
dtms = np.concatenate((dtms[0], dtms[1]), axis=2).squeeze(axis=1)
dtms.shape

def plot_solar_algorithm_performance(df, var, figsize=(10, 10), ylims_mean=None, ylims_std=None):
    # Plot a grid where each row is a solar activity level, and each column
    # is a error type (sfu or percent). Each cell is the result of calling the 
    # function plot_fe.
    # Input:
    # df: dataframe with the results of the forecasting experiment, with the columns
    #     variable, condition, horizon, µ_sfu, σ_sfu, µ_percent, σ_percent
    # var: variable to plot (F10, S10, M10, Y10)
    # figsize: figure size
    # ylims_mean: List with the y limits of the mean for each error type: 
    #   [(percent[0], percent[1]), (sfu[0], sfu[1])]
    # ylims_std: List with the y limits of the standard deviation for each error type:
    #   [(percent[0], percent[1]), (sfu[0], sfu[1])]
    # Output:
    # None, but it plots the grid
    sals = ['low', 'moderate', 'elevated', 'high']
    fig, axs = plt.subplots(len(sals), 2, figsize=figsize)
    fig.suptitle(f"Forecast error for {var}")
    for sal_idx, sal in enumerate(sals):
        for idx, err_type in enumerate(['percent', 'sfu']):
            df_var = df[(df['variable'] == var) & (df['condition'] == sal)]
            # Minimum and maximum values across column
            min_val_mean = df[f'µ_{err_type}'].min() if ylims_mean is None else ylims_mean[idx][0]
            max_val_mean = df[f'µ_{err_type}'].max() if ylims_mean is None else ylims_mean[idx][1]
            min_val_std = df[f'σ_{err_type}'].min() if ylims_std is None else ylims_std[idx][0]
            max_val_std = df[f'σ_{err_type}'].max() if ylims_std is None else ylims_std[idx][1]

            mean_fe = df_var[f'µ_{err_type}'].values
            std_fe = df_var[f'σ_{err_type}'].values
            ax = axs[sal_idx, idx]
            ax.plot(mean_fe, color='#000000')
            ax.set_xlabel('Days from Epoch')
            ax.set_ylabel(f'Mean [{err_type}]', color='#000000')
            ax.tick_params(axis='y', labelcolor='#000000')
            ax.set_xticks(range(len(mean_fe)))
            ax.set_xticklabels(range(1, len(mean_fe)+1))
            ax.set_ylim(min_val_mean, max_val_mean)
            ax2 = ax.twinx()
            ax2.plot(std_fe, color='tab:red')
            ax2.set_ylabel(f'STD [{err_type}]', color='tab:red')
            ax2.set_ylim(min_val_std, max_val_std)
            ax2.tick_params(axis='y', labelcolor='tab:red')
            # ax2.set_xticks(range(len(std_fe)))
            # ax2.set_xticklabels(range(1, len(std_fe)+1))
            n_samples = df_var['n_samples'].values[0] 
            ax.set_title(f'{sal}\n{n_samples} forecasts')
            # Draw a grid in the background
            ax.grid(True, which='both', axis='both', color='lightgrey',
                    linestyle='-', linewidth=0.5)
            fig.tight_layout()