import numpy as np
from tqdm import tqdm
from .utils import load_json, get_h5_files, load_from_hdf5
from .triggers import get_triggerstamps, get_triggertimes_ms, timeshift_triggerstamps
# from get_masked_data import get_baseline_subtracted_traces, get_masked_ON_OFF_traces
import pandas as pd

def get_averages_based_on_condition(Exp: object, conditions: list, led_nm: list) -> tuple:
    """
    Function to get the averaged traces, trigger stamps, ticks, and baselines based on the conditions
    and LEDs for analysis.

    Parameters:
        Exp (object): Exp object containing the data for the experiment
        conditions (list): List of conditions for the experiment
        led_nm (list): List of LEDs for analysis

    Returns:
        avgs (np.array): Averaged traces
        trigger_stamps (np.array): Trigger stamps [ms] for each LED
        ticks (np.array): Ticks [ms] for each snippet
        baselines (np.array): Baselines for the trace before the first trigger
    """
    nleds = len(conditions)

    # Get the ticks [ms] for each snippet
    ticks = Exp.SnippetsTimes0[:, 0, :].copy()

    # Get the trigger times [ms]
    trigger_times = get_triggertimes_ms(Exp)

    # Get the trigger stamps [ms] for each LED
    trigger_stamps = get_triggerstamps(trigger_times, nleds)

    # Time shift the ticks and trigger stamps
    tshift = Exp.SnippetsTimes0[:, 0, 0].reshape(-1, 1)
    ticks -= tshift
    trigger_stamps_shifted = timeshift_triggerstamps(trigger_stamps, tshift[0])

    # Get the averaged traces
    avgs = Exp.Averages0.copy()

    # List to store the averaged traces to concatenate
    avgs_temp = []
    # List to store the ticks to concatenate
    ticks_temp = []

    # Iterate over the conditions
    for led, led_idx in conditions.items():
        # If the LED is in the list of LEDs for analysis
        if led in led_nm:
            # Get the tick [ms] for the LED when ON
            tick_on = np.where(ticks == np.round(trigger_stamps_shifted[led_idx][0, 0]))[1][0]

            # If it's the first LED
            if led_idx == 0:
                # Get the baseline for the trace before the first trigger
                baselines = avgs[:, :tick_on]

            # If it's the last LED
            if led_idx == nleds - 1:
                # Get the tick [ms] for the next LED when ON
                tick_off = np.where(ticks == np.round(trigger_stamps_shifted[led_idx][0, 1]))[1][0]

                # TODO: Clean up A LOT, there must be an easier way to generalize this
                tick_off += tick_off - tick_on
            else:
                # Get the tick [ms] for the next LED when ON
                tick_off = np.where(ticks == np.round(trigger_stamps_shifted[led_idx + 1][0, 0]))[1][0]

            # Append the averaged traces and ticks to the lists
            avgs_temp.append(avgs[:, tick_on:tick_off])
            ticks_temp.append(ticks[:, tick_on:tick_off])

    # Concatenate the averaged traces and ticks
    avgs = np.concatenate(avgs_temp, axis=1)
    ticks = np.concatenate(ticks_temp, axis=1)

    return avgs, trigger_stamps, ticks, baselines

def get_avg_traces_from_exp(Exp: object, exp_info_path: str) -> tuple:
    """
    Function to get the averaged traces, trigger stamps, ticks, and baselines from an Exp object.
    
    The function checks which type of experiment it is (led6, led7, or led8) and then calls the
    get_avgs_based_on_condition function to get the averaged traces, trigger stamps, ticks, and baselines.
    
    Parameters:
        Exp (object): Exp object containing the data for the experiment
    
    Returns:
        avgs (np.array): averaged traces
        trigger_stamps (np.array): trigger stamps
        ticks (np.array): ticks
        baselines (np.array): baselines
    """
    conditions = load_json(r'..\experiment_info\zf_led_info.json')
    led_nm = load_json(exp_info_path)
    
    # Check which type of experiment it is
    #TODO write: for condition in list of conditions - new JSON for [{{'led6} : {}}, {{'led7': {}}, {'led8': {}}}]
    if 'led6' in Exp.filename:
        # If it's an led6 experiment, get the averaged traces, trigger stamps, ticks, and baselines
        # using the conditions for led6
        avgs, trigger_stamps, ticks, baselines = get_averages_based_on_condition(Exp, conditions[2], led_nm)            
    elif 'led7' in Exp.filename:
        # If it's an led7 experiment, get the averaged traces, trigger stamps, ticks, and baselines
        # using the conditions for led7
        avgs, trigger_stamps, ticks, baselines = get_averages_based_on_condition(Exp, conditions[1], led_nm)
    elif 'led8' in Exp.filename:
        # If it's an led8 experiment, get the averaged traces, trigger stamps, ticks, and baselines
        # using the conditions for led8
        avgs, trigger_stamps, ticks, baselines = get_averages_based_on_condition(Exp, conditions[0], led_nm)        
    
    return avgs, trigger_stamps, ticks, baselines


def movingaverage_to_pad(ar: np.array, npads: int, win_size: int) -> np.array:
    """
    Extends an array of traces by applying a moving average to the last 'npads' elements.
    
    Args:
        ar (np.array): array of traces to be extended
        npads (int): number of elements to be extended
        win_size (int): size of the moving average window
    
    Returns:
        np.array: extended array of traces
    """
    
    # Calculate the difference between npads and win_size
    if npads-win_size < 0:
        diff = npads + win_size
    else:
        diff = npads

    # Create a pandas DataFrame from the last 'diff' elements of the array
    df = pd.DataFrame(ar[:, -diff-1:-1])
    
    # Apply a rolling mean to the DataFrame with a window size of 'win_size'
    windows = df.T.rolling(win_size)
    
    # Calculate the mean of the rolling window and select the last 'npads' rows
    mns = windows.mean().to_numpy()[-npads:, :]
    
    # Return the extended array
    return mns.T


def get_extended_traces(traces: np.array, npads: int, win_size: int = 3) -> np.array:
    """
    Extends an array of traces by applying a moving average to the last 'npads' elements.
    
    Args:
        traces (np.array): array of traces to be extended
        npads (int): number of elements to be extended
        win_size (int, optional): size of the moving average window. Defaults to 3.
    
    Returns:
        np.array: extended array of traces
    """
    # Apply a moving average to the last 'npads' elements of the array
    extend_ar = movingaverage_to_pad(traces, npads, win_size=win_size)
    
    # Concatenate the original array with the extended array
    extended_traces = np.concatenate((traces, extend_ar), axis=1)
    
    return extended_traces

def pool_avg_traces(dir: str, exp_info_path: str, nLEDs: int):
    """
    Function to pool all the avg traces with snippets contaminated with light artifacts masked
    
    Parameters:
    dir (str): directory path containing the .h5 files
    nLEDs (int): number of LEDs in the experiment
    
    Returns:
    pooled_avg_traces (np.array): pooled avg traces with snippets contaminated with light artifacts masked
    pooled_triggerstamps (np.array): pooled trigger stamps
    pooled_ticks (np.array): pooled ticks
    pooled_baselines (np.array): pooled baselines
    quality_criterion_all (np.array): quality criterion for each snippet
    """
    # Get all the h5 files from the directory
    files = get_h5_files(dir)
    
    # Initialize variables to store the pooled data
    # Iterate over each h5 file
    for i, fpath in tqdm(enumerate(files)):
        # Load the Exp object from the h5 file
        Exp = load_from_hdf5(fpath)
        
        # Get the avg traces, trigger stamps, ticks, and baselines from the Exp object
        avgs, trigger_stamps, ticks, baselines = get_avg_traces_from_exp(Exp, exp_info_path)
        
        # If it's the first file, initialize the variables with the avgs
        if i == 0:
            pooled_avg_traces = avgs
            pooled_triggerstamps = trigger_stamps
            pooled_ticks = ticks                     
            quality_criterion_all = Exp.QualityCriterion.reshape(-1, 1)
            pooled_baselines = baselines
        else:
            # Calculate the number of pads required
            npads = int(avgs.shape[1] - pooled_avg_traces.shape[1])
            
            # If there are pads, append the avgs to the pooled_avg_traces
            if npads > 0:
                pooled_avg_traces = np.append(pooled_avg_traces, avgs[:, :-npads], axis=0)
                pooled_ticks = np.append(pooled_ticks, ticks[:, :-npads], axis=0)   
            
            # If there are negative pads, extend the avgs and ticks before appending
            if npads < 0:
                avgs = get_extended_traces(avgs, npads=np.abs(npads), win_size=3)               
                ticks = get_extended_traces(ticks, npads=np.abs(npads), win_size=3)
                
                pooled_avg_traces = np.append(pooled_avg_traces, avgs, axis=0)
                pooled_ticks = np.append(pooled_ticks, ticks[:, :pooled_ticks.shape[1]], axis=0)
            else:
                pooled_avg_traces = np.append(pooled_avg_traces, avgs, axis=0)
                pooled_ticks = np.append(pooled_ticks, ticks, axis=0)
            
            # Append the quality criterion to the quality_criterion_all
            quality_criterion_all = np.append(quality_criterion_all, Exp.QualityCriterion.reshape(-1, 1), axis=0)
            
            # Append the baselines to the pooled_baselines
            pooled_baselines = np.append(pooled_baselines, baselines, axis=0)
    
    # Print the final shape of pooled traces
    print('final shape of pooled traces:', pooled_avg_traces.shape[0], 'rois |', pooled_avg_traces.shape[1], ' time points')
    
    # Return the pooled data
    return pooled_avg_traces, pooled_triggerstamps, pooled_ticks, pooled_baselines, quality_criterion_all


def get_baseline_subtracted_traces(pooled_traces: tuple, baselines: np.array, snippets: bool):
    """
    Subtract baselines from pooled traces.

    Args:
        pooled_traces (tuple): Tuple containing pooled ON and pooled OFF traces.
        baselines (np.array): Array containing baseline values.
        snippets (bool): Flag indicating whether to return snippets.

    Returns:
        tuple: Baseline subtracted ON and OFF traces.
    """

    # If snippets flag is True, return baseline subtracted ON and OFF traces for each snippet
    if snippets:
        pooled_ON, pooled_OFF = pooled_traces
        
        # Calculate mean baseline values for each snippet
        mean_bs = np.mean(baselines, axis=1)
        
        # Reshape mean_bs to match pooled traces shape
        mean_bs = mean_bs[:, np.newaxis, np.newaxis]
        
        # Subtract mean baseline values from pooled traces
        return pooled_ON - mean_bs, pooled_OFF - mean_bs
    
    # If snippets flag is False, return baseline subtracted pooled traces
    else:
        # Calculate mean baseline values for each ROI
        mean_bs = np.mean(baselines, axis=1)
        
        # Reshape mean_bs to match pooled traces shape
        mean_bs = mean_bs[:, np.newaxis]
        
        # Subtract mean baseline values from pooled traces
        return pooled_traces - mean_bs

def get_downsampled_data(traces: np.array, num_of_templates_ticks: int) -> np.array:
    """Downsamples the input data to the desired number of ticks.
    
    Args:
        traces (np.array): The input data to be downsampled.
        num_of_templates_ticks (int): The desired number of ticks in the downsampled data.
        
    Returns:
        np.array: The downsampled data.
    """
    # Calculate the downsampling factor to achieve the desired number of ticks
    sampling_f = int(np.round((traces.shape[1] / num_of_templates_ticks)))
    
    # Downsample the data using the decimate function from scipy.signal
    ds_traces = sig.decimate(traces, sampling_f, n=8, axis=1)
    
    # Return the downsampled data
    return ds_traces