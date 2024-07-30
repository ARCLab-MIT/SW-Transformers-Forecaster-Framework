# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['config', 'plot_forecast_2', 'plot_solar_algorithm_performance', 'plot_solar_algorithm_performance_comparison',
           'plot_solar_algorithm_performance_all_indices', 'bold_best', 'convert_uuids_to_indices',
           'create_latex_comparison_tables', 'get_classified_columns', 'euclidean_distance_dict',
           'find_closest_distribution', 'sliding_window_generator', 'download_dst_data', 'generate_preprocessed_data',
           'run_optuna_study', 'filter_nb']

# %% ../nbs/utils.ipynb 2
import numpy as np
import pandas as pd
import os
from fastcore.all import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tsai.basics import SlidingWindow
from tsai.utils import yaml2dict, load_object
from tsai.data.external import download_data
from collections import Counter
from itertools import combinations, chain
import more_itertools as mit
from tqdm import tqdm
import requests
import papermill as pm
import nbformat
import sys
sys.path.append('../dev_nbs')



config = yaml2dict('../dev_nbs/config/solfsmy.yaml', attrdict=True)

# %% ../nbs/utils.ipynb 3
def plot_forecast_2(X_true, y_true, y_pred, dtms=None, sel_vars=None, idx=None, figsize=(8, 4), n_samples=1):
    #TODO: add support for dynamic x axis interval in set_major_locator
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def _plot_forecast(X_true, y_true, y_pred, sel_var=None, idx=None, figsize=(8, 4)):
        if idx is None:
            idx = np.random.randint(0, len(X_true))
        if sel_var is None:
            title = f'sample: {idx}'
        else:
            title = f'sample: {idx} sel_var: {sel_var}'
        if sel_var is None: sel_var = slice(None)

        pred = np.concatenate([X_true[idx, sel_var], y_true[idx, sel_var]], -1)
        pred[..., :X_true.shape[-1]] = np.nan

        true = np.concatenate([X_true[idx, sel_var], y_pred[idx, sel_var]], -1)
        true_hist = true.copy()
        true_fut = true.copy()

        true_hist[..., X_true.shape[-1]:] = np.nan
        true_fut[..., :X_true.shape[-1]] = np.nan
                
        plt.figure(figsize=figsize)
        if dtms is not None:
            #dtms_plot = pd.to_datetime(dtms[idx])
            dtms_plot = mdates.date2num(dtms[idx])
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
            plt.xlim(min(dtms_plot), max(dtms_plot))
            plt.plot(dtms_plot, pred.T, color='orange', lw=1, linestyle='--')
            plt.plot(dtms_plot, true_hist.T, color='purple', lw=1)
            plt.plot(dtms_plot, true_fut.T, color='purple', lw=1, linestyle='--')
            plt.axvline(dtms_plot[X_true.shape[-1]-1], color='gray', lw=.5, linestyle='--')
        else:
            plt.xlim(0, X_true.shape[-1] + y_true.shape[-1])
            plt.plot(pred.T, color='orange', lw=1, linestyle='--')
            plt.plot(true_hist.T, color='purple', lw=1)
            plt.plot(true_fut.T, color='purple', lw=1, linestyle='--')
            plt.axvline(X_true.shape[-1] - 1, color='gray', lw=.5, linestyle='--')

        plt.title(title)
        pred_patch = mpatches.Patch(color='orange', label='pred')
        true_patch = mpatches.Patch(color='purple', label='true')
        plt.legend(handles=[true_patch, pred_patch], loc='best')
        plt.show()
      
    assert X_true.shape[:-1] == y_true.shape[:-1] == y_pred.shape[:-1]
    assert y_true.shape[-1] == y_pred.shape[-1]
    
    if idx is not None:
        idx = listify(idx)
        n_samples = len(idx)
        iterator = idx
    else:
        iterator = random_randint(len(X_true), size=n_samples)
    
    if sel_vars is None:
        for idx in iterator:
            _plot_forecast(X_true, y_true, y_pred, sel_var=None, idx=idx, figsize=figsize)
    else:
        for idx in iterator:
            if sel_vars is True:
                sel_vars = np.arange(y_true.shape[1])
            else:
                sel_vars = listify(sel_vars)
            for sel_var in sel_vars:
                _plot_forecast(X_true, y_true, y_pred, sel_var=sel_var, idx=idx, figsize=figsize)


# %% ../nbs/utils.ipynb 5
def plot_solar_algorithm_performance(df, var, figsize=(10, 10), ylims=None):
    # Plot a grid where each row is a solar activity level, and each column
    # is a error type (sfu or percent). Each cell is the result of calling the 
    # function plot_fe.
    # Input:
    # df: dataframe with the results of the forecasting experiment, with the columns
    #     variable, condition, horizon, mean_sfu, std_sfu, mean_percent, std_percent
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
            min_val = df[f'mean_{err_type}'].min() - np.abs(df[f'std_{err_type}'].max()) if ylims is None else ylims[idx][0]
            max_val = df[f'mean_{err_type}'].max() + np.abs(df[f'std_{err_type}'].max()) if ylims is None else ylims[idx][1]


            mean_fe = df_var[f'mean_{err_type}'].values
            std_fe = df_var[f'std_{err_type}'].values
            ax = axs[sal_idx, idx]
            p1 = ax.plot(mean_fe, color='#000000', label='Mean')
            ax.set_xlabel('Days from Epoch')
            ax.set_ylabel(f'Mean [{err_type}]', color='#000000')
            ax.tick_params(axis='y', labelcolor='#000000')
            ax.set_xticks(range(len(mean_fe)))
            ax.set_xticklabels(range(1, len(mean_fe)+1))
            ax.set_ylim(min_val, max_val)
            ax.set_yticks(np.arange(-20, 21, 5))
            ax.fill_between(range(len(mean_fe)), mean_fe - np.abs(std_fe), mean_fe + np.abs(std_fe), color='red', alpha=0.2)

            p2 = ax.fill(np.NaN, np.NaN, 'red', alpha=0.5)
            ax.legend([p1[0], p2[0]], ['Mean Error', '[STDE]'], loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)
            
            n_samples = df_var['n_samples'].values[0] 
            ax.set_title(f'{sal}\n{n_samples} forecasts', pad=15)
            # Draw a grid in the background
            ax.grid(True, which='both', axis='both', color='lightgrey',
                    linestyle='-', linewidth=0.5)
            fig.tight_layout()

# %% ../nbs/utils.ipynb 7
def plot_solar_algorithm_performance_comparison(df, df_paper, var, figsize=(10, 10), ylims=None):
    """
    Plots and compares forecast error metrics from two datasets for a given variable across different solar activity levels.

    Parameters:
    - df (DataFrame): Main dataset containing error metrics and sample counts.
    - df_paper (DataFrame): Benchmark dataset for comparison.
    - var (str): Variable of interest (e.g., temperature).
    - figsize (tuple): Dimensions of the figure (width, height).
    - ylims (list of tuples): Optional y-axis limits for each plot; defaults to auto-calculated based on data.

    This function creates a grid of subplots for each solar activity level, comparing two error types. 
    It adjusts y-limits, annotates subplots with sample sizes, and adds a legend and grid for clarity.
    """

    sals = ['low', 'moderate', 'elevated', 'high']
    fig, axs = plt.subplots(len(sals), 2, figsize=figsize)
    fig.suptitle(f"Forecast error for {var}")
    for sal_idx, sal in enumerate(sals):
        for idx, err_type in enumerate(['percent','sfu']):
            df_var = df[(df['variable'] == var) & (df['condition'] == sal)]
            df_paper_var = df_paper[(df_paper['variable'] == var) & (df_paper['condition'] == sal)]

            # Minimum and maximum values across column
            min_val = df_var[f'mean_{err_type}'].min() - np.abs(df_var[f'std_{err_type}'].max()) if ylims is None else ylims[idx][0]
            max_val = df_var[f'mean_{err_type}'].max() + np.abs(df_var[f'std_{err_type}'].max()) if ylims is None else ylims[idx][1]
            
            min_val_paper = df_paper_var[f'mean_percent'].min() - np.abs(df_paper_var[f'std_percent'].max()) if ylims is None else ylims[idx][0]
            max_val_paper = df_paper_var[f'mean_percent'].max() + np.abs(df_paper_var[f'std_percent'].max()) if ylims is None else ylims[idx][1]
            
            if np.isnan(min_val) or np.isnan(max_val):
                min_val = min_val_paper
                max_val = max_val_paper
            
            min_val = min(min_val, min_val_paper)
            min_val = int(min_val) - (int(min_val)%10)
            max_val = max(max_val, max_val_paper)
            max_val = int(max_val) + (10 - int(max_val)%10)

            mean_fe = df_var[f'mean_{err_type}'].values
            mean_fe_paper = df_paper_var[f'mean_{err_type}'].values
            std_fe = df_var[f'std_{err_type}'].values
            std_fe_paper = df_paper_var[f'std_{err_type}'].values 
            ax = axs[sal_idx, idx]
            p1 = ax.plot(mean_fe, color='#c00000')
            p3 = ax.plot(mean_fe_paper, color='#572364')

            ax.set_xlabel('Days from Epoch')
            ax.set_ylabel(f'Mean [{err_type}]', color='#000000')

            ax.tick_params(axis='y', labelcolor='#000000')
            ax.set_xticks(range(len(mean_fe)))
            ax.set_xticklabels(range(1, len(mean_fe)+1))
            ax.set_ylim(min_val, max_val)
            steps = (max_val-min_val)/10
            ax.set_yticks(np.arange(min_val, max_val,steps))
            ax.fill_between(range(len(mean_fe)), mean_fe - np.abs(std_fe),
                            mean_fe + np.abs(std_fe),
                            color='#ff8000', alpha=0.3)

            ax.fill_between(range(len(mean_fe_paper)), mean_fe_paper - np.abs(std_fe_paper),
                            mean_fe_paper + np.abs(std_fe_paper),
                            color='#9a2edb', alpha=0.2)

            
            p2 = ax.fill(np.NaN, np.NaN, '#ff8000', alpha=0.3)
            p4 = ax.fill(np.NaN, np.NaN, '#9a2edb', alpha=0.2)

            ax.legend([(p1[0], p2[0]),(p3[0], p4[0])],
                        ['Mean Error[STDE]', 'Mean Error (Benchmark) [STDE]'], 
                        loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, 
                        fancybox=True, shadow=True)

            
            n_samples = df_var['n_samples'].values[0] 
            ax.set_title(f'{sal}\n{n_samples} forecasts', pad=15)
            # Draw a grid in the background
            ax.grid(True, which='both', axis='both', color='lightgrey',
                    linestyle='-', linewidth=0.5)
            fig.tight_layout()

# %% ../nbs/utils.ipynb 8
def plot_solar_algorithm_performance_all_indices(df, df_paper, figsize=(40, 15), ylims=None, save_fig=False):
    # Plot a grid where each row is a solar activity level, and each column is a variable.
    # Each cell contains the percent error comparison.
    # Input:
    # df: DataFrame with the results of the forecasting experiment, with the columns
    #     variable, condition, horizon, mean_percent, std_percent
    # df_paper: DataFrame with the results from the paper for comparison
    # figsize: figure size
    # ylims: List with the y limits for percent error: [(percent_min, percent_max)]
    # Output: None, but it plots the grid
    
    sals = ['low', 'moderate', 'elevated', 'high']
    vars = ['F10', 'S10', 'M10', 'Y10']
    
    fig, axs = plt.subplots(len(sals), len(vars), figsize=figsize)
    
    for sal_idx, sal in enumerate(sals):
        for var_idx, var in enumerate(vars):
            df_var = df[(df['variable'] == var) & (df['condition'] == sal)]
            df_paper_var = df_paper[(df_paper['variable'] == var) & (df_paper['condition'] == sal)]

            # Determine min and max values for y-axis
            min_val = df_var['mean_percent'].min() - np.abs(df_var['std_percent'].max()) if ylims is None else ylims[0][0]
            max_val = df_var['mean_percent'].max() + np.abs(df_var['std_percent'].max()) if ylims is None else ylims[0][1]
            
            min_val_paper = df_paper_var['mean_percent'].min() - np.abs(df_paper_var['std_percent'].max()) if ylims is None else ylims[0][0]
            max_val_paper = df_paper_var['mean_percent'].max() + np.abs(df_paper_var['std_percent'].max()) if ylims is None else ylims[0][1]
            
            min_val = min(min_val, min_val_paper)
            min_val = int(min_val) - (int(min_val)%10)
            max_val = max(max_val, max_val_paper)
            max_val = int(max_val) + (10 - int(max_val)%10)

            mean_fe = df_var['mean_percent'].values
            mean_fe_paper = df_paper_var['mean_percent'].values
            std_fe = df_var['std_percent'].values
            std_fe_paper = df_paper_var['std_percent'].values 
            
            ax = axs[sal_idx, var_idx]
            p1, = ax.plot(mean_fe, color='#c00000', lw=2.7)
            p2, = ax.plot(mean_fe_paper, color='#572364', lw=2.7)

            if sal_idx == len(sals) - 1:
                ax.set_xlabel('Days from Epoch', fontsize=18)
            if var_idx == 0:
                ax.set_ylabel(f'Error [%]', color='#000000', fontsize=18)

            ax.tick_params(axis='y', labelcolor='#000000')
            ax.set_xticks(range(len(mean_fe)))
            ax.set_xticklabels(range(1, len(mean_fe)+1), fontsize=14)
            ax.set_ylim(min_val, max_val)
            steps = (max_val - min_val) / 10
            ax.set_yticks(np.arange(min_val, max_val, steps))
            ax.set_yticklabels(np.arange(min_val, max_val, steps), fontsize=14)
            p3 = ax.fill_between(range(len(mean_fe)), mean_fe - np.abs(std_fe),
                            mean_fe + np.abs(std_fe),
                            color='#ff8000', alpha=0.3)

            p4 = ax.fill_between(range(len(mean_fe_paper)), mean_fe_paper - np.abs(std_fe_paper),
                            mean_fe_paper + np.abs(std_fe_paper),
                            color='#9a2edb', alpha=0.2)

            n_samples = df_var['n_samples'].values[0] 
            if sal_idx == 0:
                ax.set_title(f'{var}.7' , fontweight='bold', fontsize=26, pad=35)
       
            ax.text(0, max_val - steps, f'{n_samples} forecasts', fontsize=12, fontstyle='italic')
            ax.text(2.5, max_val + 1, f'{sal.capitalize()} Solar Activity', fontsize=20, fontstyle='italic', ha='center', )



            # Draw a grid in the background
            ax.grid(True, which='both', axis='both', color='lightgrey',
                    linestyle='-', linewidth=0.5)

    # Create a single legend at the bottom
    example_mean = ax.plot([], [], color='grey', lw=2.5, label='Mean')[0]
    example_std = ax.fill_between([], [], [], color='grey', alpha=0.3)

    # Combine handles and labels
    handles_example = [example_mean, example_std]
    labels_example = ['Mean', 'STD']
    measures_legend = fig.legend(handles_example, labels_example, 
                                loc='upper center', ncol=1, 
                                fancybox=True, shadow=True, fontsize=18, 
                                bbox_to_anchor=(0.47, 0.06), title='Measures'
                                )
    measures_legend.get_title().set_fontsize('14')

    handles = [(p1, p3), (p2, p4)]
    labels = ['PatchTST', 'SOLAR2000']

    algorithm_legend = fig.legend(handles, labels, 
                                  loc='lower center', ncol=1, 
                                  fancybox=True, shadow=True, fontsize=18, 
                                  bbox_to_anchor=(0.53, -0.032), title='Forecast Models'
                                  )
    algorithm_legend.get_title().set_fontsize('14')
    plt.subplots_adjust(hspace=0.3) 

    if save_fig:
        plt.savefig('figures/solfsmy_comparison.png', format='png', bbox_inches='tight', dpi=300)

    plt.show()

# %% ../nbs/utils.ipynb 9
def bold_best(X, X_ref, higher_better=False, bold_ref_too=False, 
              bold_equal=False, use_abs=False):
    """
        Returns X with the best values in bold, with respect to X_ref, position by
        position, i.e., if X[0] is better than X_ref[0] it will be bolded. 
        Input:
            X: 1D numpy array
            X_ref: 1D numpy array
            higher_better: If True, then the best values are the highest ones
            bold_ref_too: If True, best values in X_ref are also in bold.
            bold_equal: If bold_equal is True, then the values equal to the 
            best ones are also bolded
            use_abs: If True, then the absolute values are used to compare
        Output:
            X: 1D numpy array with the best values in bold (or a tuple of two
            1D numpy arrays if bold_ref_too is True)
    """
    if use_abs:
        X_abs = np.abs(X)
        X_ref_abs = np.abs(X_ref)
    else:
        X_abs = X
        X_ref_abs = X_ref
    if higher_better:
        if bold_equal:
            best = np.greater_equal(X_abs, X_ref_abs)
        else:
            best = np.greater(X_abs, X_ref_abs)
    else:
        if bold_equal:
            best = np.less_equal(X_abs, X_ref_abs)
        else:
            best = np.less(X_abs, X_ref_abs)
    # Make bold
    X = np.array([f'\\textbf{{{x}}}' if best[i] else f'{x}' for i, x in enumerate(X_abs)])
    if bold_ref_too:
        X_ref = np.array([f'\\textbf{{{x}}}' if not best[i] else f'{x}' for i, x in enumerate(X_ref_abs)])
        return X, X_ref
    else:
        return X

# %% ../nbs/utils.ipynb 11
def convert_uuids_to_indices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuids = re.findall(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b", cuda_visible_devices)

    if uuids:
        indices = [str(i) for i in range(len(uuids))]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)

# %% ../nbs/utils.ipynb 12
def _highlight_better_values(df, our_col, paper_col, stat):
    return np.where(
        np.abs(df[our_col][stat]) < np.abs(df[paper_col][stat]),
        '\\textbf{' + df[our_col][stat].astype(str) + '}',
        df[our_col][stat].astype(str)
    )

def create_latex_comparison_tables(results_df, forecast_variables, forecast_horizon):
    """
    Generates LaTeX tables comparing our model's forecast results with a reference paper.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing 'variable', 'condition', 'horizon', 'statistic', 'value_ours', 'value_paper'.
    - forecast_variables (list of str): List of forecast variables to include.
    - forecast_horizon (int): Number of forecast days.

    Example:
    create_comparison_tables(results_df, ['var1', 'var2'], 1)
    """
    reshaped_df = results_df.pivot_table(
        index=['variable', 'condition', 'horizon'],
        columns='statistic',
        values=['value_ours', 'value_paper']
    )

    # Highlight better mean and std values
    reshaped_df[('value_ours', 'mean_str')] = _highlight_better_values(reshaped_df, 'value_ours', 'value_paper', 'mean')
    reshaped_df[('value_ours', 'std_str')] = _highlight_better_values(reshaped_df, 'value_ours', 'value_paper', 'std')
    reshaped_df[('value_paper', 'mean_str')] = _highlight_better_values(reshaped_df, 'value_paper', 'value_ours', 'mean')
    reshaped_df[('value_paper', 'std_str')] = _highlight_better_values(reshaped_df, 'value_paper', 'value_ours', 'std')

    # Drop the original mean and std columns
    reshaped_df = reshaped_df.drop(columns=[('value_ours', 'mean'), ('value_ours', 'std'),
                                            ('value_paper', 'mean'), ('value_paper', 'std')])

    # Combine mean and std into a single column
    reshaped_df[('value_ours', 'mean ± std')] = reshaped_df[('value_ours', 'mean_str')] + ' ± ' + reshaped_df[('value_ours', 'std_str')]
    reshaped_df[('value_paper', 'mean ± std')] = reshaped_df[('value_paper', 'mean_str')] + ' ± ' + reshaped_df[('value_paper', 'std_str')]

    # Drop the intermediate string columns
    reshaped_df = reshaped_df.drop(columns=[('value_ours', 'mean_str'), ('value_ours', 'std_str'),
                                            ('value_paper', 'mean_str'), ('value_paper', 'std_str')])

    # Clean up column levels
    reshaped_df.columns = reshaped_df.columns.droplevel(0)
    reshaped_df.columns = ['NN', 'benchmark']
    reshaped_df = reshaped_df.reset_index()

    # Sort conditions
    reshaped_df['condition'] = pd.Categorical(reshaped_df['condition'], categories=['low', 'moderate', 'elevated', 'high'], ordered=True)
    reshaped_df = reshaped_df.sort_values(by=['variable', 'condition'])

    # Print LaTeX tables for each variable
    for variable in forecast_variables:
        variable_df = reshaped_df[reshaped_df['variable'] == variable].drop(columns='variable')
        latex_table = variable_df.to_latex(
            index=False, 
            escape=False,
            column_format='|l|' + '|c|' * forecast_horizon,
            caption=f'Comparison of the results of the paper with the results of our model for the variable {variable}',
            label=f'tab:comparison_{variable}'
        )
        print(latex_table)

# %% ../nbs/utils.ipynb 14
def get_classified_columns (df: pd.DataFrame, thresholds:dict, activity_levels:dict):
    """
    Creates classified columns based on predefined ranges for specified columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with classification classification of each column.

    """
    columns_to_classify = df.columns.intersection(thresholds.keys())

    if columns_to_classify.empty:
        return df
    else:
        df_cat = pd.DataFrame()
        for column in columns_to_classify:
            # ranges tuples come as strings in the yaml file, so we need to convert them to tuples with eval
            bins = pd.IntervalIndex.from_tuples(thresholds[column])
            df_cat[f'{column}_Cat'] = np.array(activity_levels[column])[pd.cut(df[column], bins=bins).cat.codes]
        return df_cat

# %% ../nbs/utils.ipynb 16
def euclidean_distance_dict(X:dict, Y:dict):
    return math.sqrt(sum((X.get(d,0) - Y.get(d,0))**2 for d in set(X) | set(Y)))


# %% ../nbs/utils.ipynb 18
def find_closest_distribution(df_cat, target_distribution, segment_size, val_size):
    """
    Finds the combination of segments in the categorical data that is closest to the target distribution.

    Parameters:
    df_cat (pd.Series): A pandas Series containing the categorical data.
    target_distribution (dict): The target distribution to compare against, given as a dictionary where keys are categories and values are their target proportions.
    segment_size (int): The size of each segment to split the data into.
    val_size (float): The proportion of the validation split.

    Returns:
    best_combination (tuple): The indices of the segments that form the closest combination to the target distribution.
    segments (list): The list of segments created from the data.
    distribution_found (dict): The distribution of categories in the best combination of segments.
    """
    idxs = list(df_cat.index)
    segments = np.array_split(idxs, len(df_cat) // segment_size)

    value_counts = [df_cat[segments[i]].value_counts().to_dict() for i in range(len(segments))]

    num_segments = int(len(segments)*(val_size))
    print(f"Total number of segments:{ len(segments)}, Number of segments for validation: {num_segments} ({num_segments/len(segments)*100:.2f}%)")

    
    best_combination = None
    best_distance = np.inf
    distribution_found = None
    comb = combinations(range(len(value_counts)), num_segments)
    for c in tqdm(comb):
        values = Counter({})
        for i in c:
            values = values + Counter(value_counts[i])
        total = sum(values.values(), 0.0)
        distribution = {k: v / total for k, v in values.items()}
        
        distance = euclidean_distance_dict(distribution, target_distribution)

        if distance < best_distance:
            best_distance = distance
            best_combination = c
            distribution_found = distribution
    print("The closest group of segments to F10.7 categories has an euclidean distance of", best_distance)
    return best_combination, segments, distribution_found

# %% ../nbs/utils.ipynb 20
def sliding_window_generator(df, split_start, data_columns, config, comb=None, segments=None):
    consecutive_elements, X, y = None, None, None

    if comb is not None:
        consecutive_elements = [list(group) for group in mit.consecutive_groups(comb)]

        df_to_window = []
        for elements in consecutive_elements:
            best_comb_idxs = [segments[i] for i in elements]
            df_to_window.append(df.iloc[chain.from_iterable(best_comb_idxs)])
    else:
        df_to_window = [df]

    X_window, y_window = None, None 
    for df_window in df_to_window:    
        X_window, y_window = SlidingWindow(
            window_len=config.lookback,
            horizon=config.horizon, 
            stride=1, 
            get_x=data_columns, 
            get_y=data_columns
        )(df_window)
        X = np.concatenate([X, X_window]) if X is not None else X_window
        y = np.concatenate([y, y_window]) if y is not None else y_window
    
    
    splits = L(list(np.arange(split_start, len(X)+split_start)))
    return X, y, splits

# %% ../nbs/utils.ipynb 22
def download_dst_data(start_date: str = '01/1957',
                      end_date: str = pd.Timestamp.today(),
                      save_folder: str = "./dst_data"):
    """
    Downloads Dst index data between the specified start and end dates.

    :param start_date: Start date in the format 'MM/YYYY'
    :param end_date: End date in the format 'MM/YYYY'
    :param save_folder: Folder where the data files should be saved
    """

    os.makedirs(save_folder, exist_ok=True)

    # Initialize file path
    file_name = "DST_IAGA2002.txt"
    file_path = os.path.join(save_folder, file_name)


    # Remove existing file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted existing file: {file_path}")

    # Convert input dates to datetime objects
    start_dt = pd.to_datetime(start_date, format='%m/%Y')
    end_dt = pd.to_datetime(end_date, format='%m/%Y')

    # HTTP REQUEST COMPONENTS
    current_start = start_dt
    while current_start <= end_dt:
        current_end = min(current_start + pd.DateOffset(years=24), end_dt)

        # Extract year components
        SCent = current_start.year // 100
        STens = (current_start.year % 100) // 10
        SYear = current_start.year % 10
        SMonth = current_start.month

        ECent = current_end.year // 100
        ETens = (current_end.year % 100) // 10
        EYear = current_end.year % 10
        EMonth = current_end.month

        # Construct URL for current chunk
        url = f"https://wdc.kugi.kyoto-u.ac.jp/cgi-bin/dstae-cgi?" \
              f"SCent={SCent}&" \
              f"STens={STens}&" \
              f"SYear={SYear}&" \
              f"SMonth={SMonth:02d}&" \
              f"ECent={ECent}&" \
              f"ETens={ETens}&" \
              f"EYear={EYear}&" \
              f"EMonth={EMonth:02d}&" \
              "Image+Type=GIF&COLOR=COLOR&AE+Sensitivity=0&Dst+Sensitivity=0&Output=DST&Out+format=IAGA2002"

        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Connection": "keep-alive",
            "Referer": "https://wdc.kugi.kyoto-u.ac.jp/dstae/index.html"
        }

        try:
            session = requests.session()
            response = session.get(url, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses

            # Append or write to file
            mode = 'ab' if os.path.exists(file_path) else 'wb'
            with open(file_path, mode) as file:
                file.write(response.content)

            print(f"Downloaded and saved data from {current_start.strftime('%m/%Y')} to {current_end.strftime('%m/%Y')}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download data: {e}")

        # Move to the next chunk
        current_start = current_end + pd.DateOffset(days=1)

    print(f"All data downloaded and saved to {file_path}")
    return file_path


# %% ../nbs/utils.ipynb 24
def generate_preprocessed_data(config, generate_preproc_pipe=True):
    df, preproc_pipe = None, None
    try:
        df = load_object(config.df_save_path)
        if generate_preproc_pipe:
            preproc_pipe = load_object(config.preproc_pipe_save_path)

    except FileNotFoundError:
        output = './tmp/data_out.ipynb'
        print(f"{config.df_save_path} not found. Executing the notebook to generate the data...")
        
        pm.execute_notebook(config.data_nb, output)
        os.remove(output)

        df = load_object(config.df_save_path)
        if generate_preproc_pipe:
            preproc_pipe = load_object(config.preproc_pipe_save_path)

        print("Data generated successfully.")
    
    return df, preproc_pipe

# %% ../nbs/utils.ipynb 26
from pathlib import Path
from fastcore.script import *
import joblib
from importlib import import_module
import warnings
warnings.filterwarnings("ignore")

def run_optuna_study(objective, resume=None, study_type=None, multivariate=True, search_space=None, evaluate=None, seed=None, sampler=None, pruner=None, 
                     study_name=None, direction='maximize', n_trials=None, timeout=None, gc_after_trial=False, show_progress_bar=True, 
                     save_study=True, path='optuna', show_plots=True):
    r"""Creates and runs an optuna study.

    Args: 
        objective:          A callable that implements objective function.
        resume:             Path to a previously saved study.
        study_type:         Type of study selected (bayesian, gridsearch, randomsearch). Based on this a sampler will be build if sampler is None. 
                            If a sampler is passed, this has no effect.
        multivariate:       If this is True, the multivariate TPE is used when suggesting parameters. The multivariate TPE is reported to outperform 
                            the independent TPE.
        search_space:       Search space required when running a gridsearch (if you don't pass a sampler).
        evaluate:           Allows you to pass a specific set of hyperparameters that will be evaluated.
        seed:               Fixed seed used by samplers.
        sampler:            A sampler object that implements background algorithm for value suggestion. If None is specified, TPESampler is used during 
                            single-objective optimization and NSGAIISampler during multi-objective optimization. See also samplers.
        pruner:             A pruner object that decides early stopping of unpromising trials. If None is specified, MedianPruner is used as the default. 
                            See also pruners.
        study_name:         Study’s name. If this argument is set to None, a unique name is generated automatically.
        direction:          A sequence of directions during multi-objective optimization.
        n_trials:           The number of trials. If this argument is set to None, there is no limitation on the number of trials. If timeout is also set to 
                            None, the study continues to create trials until it receives a termination signal such as Ctrl+C or SIGTERM.
        timeout:            Stop study after the given number of second(s). If this argument is set to None, the study is executed without time limitation. 
                            If n_trials is also set to None, the study continues to create trials until it receives a termination signal such as 
                            Ctrl+C or SIGTERM.
        gc_after_trial:     Flag to execute garbage collection at the end of each trial. By default, garbage collection is enabled, just in case. 
                            You can turn it off with this argument if memory is safely managed in your objective function.
        show_progress_bar:  Flag to show progress bars or not. To disable progress bar, set this False.
        save_study:         Save your study when finished/ interrupted.
        path:               Folder where the study will be saved.
        show_plots:         Flag to control whether plots are shown at the end of the study.
    """
    
    try: import optuna
    except ImportError: raise ImportError('You need to install optuna to use run_optuna_study')

    # Sampler
    if sampler is None:
        if study_type is None or "bayes" in study_type.lower(): 
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=multivariate)
        elif "grid" in study_type.lower():
            assert search_space, f"you need to pass a search_space dict to run a gridsearch"
            sampler = optuna.samplers.GridSampler(search_space)
        elif "random" in study_type.lower(): 
            sampler = optuna.samplers.RandomSampler(seed=seed)
    assert sampler, "you need to either select a study type (bayesian, gridsampler, randomsampler) or pass a sampler"

    # Study
    if resume: 
        try:
            study = joblib.load(resume)
        except: 
            print(f"joblib.load({resume}) couldn't recover any saved study. Check the path.")
            return
        print("Best trial until now:")
        print(" Value: ", study.best_trial.value)
        print(" Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else: 
        study = optuna.create_study(sampler=sampler, pruner=pruner, study_name=study_name, directions=direction)
    if evaluate: study.enqueue_trial(evaluate)
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=gc_after_trial, show_progress_bar=show_progress_bar)
    except KeyboardInterrupt:
        pass

    # Save
    if save_study:
        full_path = Path(path)/f'{study.study_name}.pkl'
        full_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(study, full_path)
        print(f'\nOptuna study saved to {full_path}')
        print(f"To reload the study run: study = joblib.load('{full_path}')")

    # Plots
    if show_plots and len(study.trials) > 1:
        try: display(optuna.visualization.plot_optimization_history(study))
        except: pass
        try: display(optuna.visualization.plot_param_importances(study))
        except: pass
        try: display(optuna.visualization.plot_slice(study))
        except: pass
        try: display(optuna.visualization.plot_parallel_coordinate(study))
        except: pass

    # Study stats
    try:
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"\nStudy statistics    : ")
        print(f"  Study name        : {study.study_name}")
        print(f"  # finished trials : {len(study.trials)}")
        print(f"  # pruned trials   : {len(pruned_trials)}")
        print(f"  # complete trials : {len(complete_trials)}")
        
        print(f"\nBest trial          :")
        trial = study.best_trial
        print(f"  value             : {trial.value}")
        print(f"  best_params = {trial.params}\n")
    except:
        print('\nNo finished trials yet.')
    return study


# %% ../nbs/utils.ipynb 27
def filter_nb (path:str, skip_tags:list):
    """
    Filters out cells with tags in skip_tags from a notebook.

    Args:
    path (str): Path to the notebook file.
    skip_tags (list): List of tags to filter out.

    Returns:
    nb (nbformat.NotebookNode): The filtered notebook.
    """
    nb = nbformat.read(path, as_version=4)

    filtered_cells = [cell for cell in nb.cells if not set(skip_tags) & set(cell.metadata.get('tags', []))]
    nb.cells = filtered_cells
    
    return nb
