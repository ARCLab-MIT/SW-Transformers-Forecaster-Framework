# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['get_idxs_per_solar_activity_level', 'convert_uuids_to_indices', 'get_classified_columns', 'euclidean_distance_dict',
           'find_closest_distribution', 'sliding_window_generator', 'download_dst_data', 'generate_preprocessed_data',
           'run_optuna_study', 'filter_nb', 'create_vectorized_condition_function']

# %% ../nbs/utils.ipynb 2
import numpy as np
import pandas as pd
from fastcore.all import *
from tsai.basics import SlidingWindow
from tsai.utils import load_object
from collections import Counter
from itertools import combinations, chain
import more_itertools as mit
from tqdm import tqdm
import requests
import papermill as pm
import nbformat

# %% ../nbs/utils.ipynb 3
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

# %% ../nbs/utils.ipynb 5
def convert_uuids_to_indices():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuids = re.findall(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b", cuda_visible_devices)

    if uuids:
        indices = [str(i) for i in range(len(uuids))]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)

# %% ../nbs/utils.ipynb 7
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

# %% ../nbs/utils.ipynb 9
def euclidean_distance_dict(X:dict, Y:dict):
    return math.sqrt(sum((X.get(d,0) - Y.get(d,0))**2 for d in set(X) | set(Y)))


# %% ../nbs/utils.ipynb 11
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

# %% ../nbs/utils.ipynb 13
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

# %% ../nbs/utils.ipynb 15
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


# %% ../nbs/utils.ipynb 17
def generate_preprocessed_data(config, generate_preproc_pipe=True, generate_exp_pipe=True):
    result = []
    try:
        df = load_object(config.df_save_path)
 
    except FileNotFoundError:
        output = './tmp/data_out.ipynb'
        print(f"{config.df_save_path} not found. Executing the notebook to generate the data...")
        
        pm.execute_notebook(config.data_nb, output)
        os.remove(output)

        print("Data generated successfully.")

    results = [load_object(config.df_save_path)]

    if generate_preproc_pipe:
        results.append(load_object(config.preproc_pipe_save_path))

    if generate_exp_pipe:
        results.append(load_object(config.exp_pipe_save_path))

    return *results,

# %% ../nbs/utils.ipynb 19
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


# %% ../nbs/utils.ipynb 20
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

# %% ../nbs/utils.ipynb 21
def create_vectorized_condition_function(geo_thresholds_dict, sol_thresholds_dict, geo_levels_dict, solact_levels_dict):
    """
    Creates a vectorized function that returns the condition based on geomagnetic and solar values.
    
    Input:
    geo_thresholds_dict: Dictionary of thresholds for the geomagnetic indices (e.g., 'AP', 'DST')
    sol_thresholds_dict: Dictionary of thresholds for the solar indices (e.g., 'F10', 'S10')
    geo_levels_dict: Dictionary of activity levels for geomagnetic indices
    solact_levels_dict: Dictionary of activity levels for solar indices
    
    Output:
    A vectorized function that can be used to compute conditions for arrays of geo_values and sol_values.
    """
    
    def get_combined_condition(geo_index, geo_value, sol_index, sol_value):
        """
        Function that returns the condition based on the geo_value and sol_value.
        Input:
            geo_index: The name of the geomagnetic index (e.g., 'AP', 'DST')
            geo_value: Value of the geomagnetic index (float)
            sol_index: The name of the solar index (e.g., 'F10', 'S10')
            sol_value: Value of the solar index (float)
        Output:
            condition: Condition string that combines geomagnetic and solar conditions (string)
        """
        geo_thresholds = geo_thresholds_dict[geo_index]
        sol_thresholds = sol_thresholds_dict[sol_index]
        geo_levels = geo_levels_dict[geo_index]
        solact_levels = solact_levels_dict[sol_index]

        geo_condition = None
        sol_condition = None

        # Determine geomagnetic condition
        for i, (lower, upper) in enumerate(geo_thresholds):
            if lower < geo_value <= upper:
                geo_condition = geo_levels[i]
                break

        # Determine solar condition
        for j, (lower, upper) in enumerate(sol_thresholds):
            if lower < sol_value <= upper:
                sol_condition = solact_levels[j]
                break

        # Combine conditions
        if geo_condition and sol_condition:
            if geo_index == 'AP':
                return f'{geo_condition}Geo_{sol_condition.capitalize()}'
            else: 
                return f'{geo_condition}_{sol_condition.capitalize()}'
        else:
            return 'Unknown'  # Fallback if no condition is found

    # Return the vectorized version of the get_combined_condition function
    return np.vectorize(get_combined_condition)



