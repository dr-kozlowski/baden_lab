import os
import json
import h5py
from tqdm.auto import tqdm
import warnings
import datetime

from dataclasses import dataclass, field
from dacite import from_dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.signal as sig
import scipy.interpolate as interp
import scipy.optimize as optimize

def load_json(fpath: str) -> list:
    """
    Load a JSON file from the given file path.

    Args:
        fpath (str): The path to the JSON file.

    Returns:
        list: The list of data loaded from the JSON file.
    """
    # Open the JSON file
    with open(fpath) as f:
        # Load the JSON data into a list
        data = json.load(f)
    
    # Close the file
    
    # Return the loaded data
    return data

def get_h5_files(dir: str) -> list:
    """
    Function to retrieve all .h5 files in a directory and its subdirectories.

    Args:
        dir (str): Directory to search for .h5 files

    Returns:
        List[str]: List of paths to .h5 files
    """
    # Use os.walk to traverse the directory and its subdirectories
    files = [os.path.join(root, file) for root, _, files in os.walk(dir) for file in files if file.endswith('.h5')]

    return files


def load_from_hdf5(path):
    """
    Load an HDF5 file directly and write its contents to a DataClass object with keys in the HDF5 file
    becoming attributes of that object.
    author: Simen


    Args:
        path (str): The path to the HDF5 file.

    Returns:
        Data_hdf5: A DataClass object containing the contents of the HDF5 file.
    Note that you don't get any of the fancy processing attributes with this, just access to waves,
    to be used only for utility 
    """

    # Create an empty dictionary
    new_dict = {}

    # Open the HDF5 file
    with h5py.File(path) as HDF5_file:

        # Add the metadata to the dictionary
        metadata = metadata_dict(HDF5_file)
        for key in HDF5_file.keys():
            new_dict[key] = np.array(HDF5_file[key]).T ## note rotation

    # Merge the data and metadata into a single dictionary
    data_dict = new_dict
    final_dict = (data_dict | metadata)

    # Create a DataClass that automatically maps contents of HDF5 file
    @dataclass
    class Data_hdf5:
        # Automatically maps contents of HDF5 file
        __annotations__ = {key: type(data_type) for key, data_type in final_dict.items()}
        def attributes(self):
            """Return a list of attribute names"""
            return list(self.__annotations__)

    # Load the dictionary into a DataClass object
    object = from_dict(Data_hdf5, final_dict)

    return object

"""Helper functions for Data classes: from: https://github.com/simbru/RF_project/blob/main/data_helpers.py """
def load_wDataCh0(HDF5_file):
    """
    Load either detrended or raw wDataCh0 from HDF5 file.

    Args:
        HDF5_file (h5py.File): HDF5 file object.

    Returns:
        numpy.ndarray or None: wDataCh0 as a numpy array with shape (time, y, x).
                              If wDataCh0 or wDataCh0_detrended cannot be found, return None.
    """
    # Prioritise detrended (because corrections applied in pre-proc)
    # Try loading detrended wDataCh0
    if "wDataCh0_detrended" in HDF5_file.keys():
        img = HDF5_file["wDataCh0_detrended"]
    # If not found, try loading raw wDataCh0
    elif "wDataCh0" in HDF5_file.keys():
        img = HDF5_file["wDataCh0"]
    # If neither can be found, raise a warning and return None
    else:
        warnings.warn("wDataCh0 or wDataCh0_detrended could not be identified. Returning None")
        img = None
    # Transpose the loaded image to have shape (time, y, x)
    return np.array(img).transpose(2,1,0)

def metadata_dict(HDF5_file):
    """
    Extract metadata from the HDF5 file.

    Args:
        HDF5_file (h5py.File): The HDF5 file object.

    Returns:
        dict: A dictionary containing the extracted metadata.
            - filename (str): The name of the HDF5 file.
            - exp_date (datetime.date): The date of the experiment.
            - exp_time (datetime.time): The time of the experiment.
            - objectiveXYZ (tuple): The objective XYZ coordinates.
    """
    # Extract experiment date and time
    date, time = get_experiment_datetime(HDF5_file["wParamsStr"])

    # Extract objective XYZ coordinates
    objectiveXYZ = get_rel_objective_XYZ(HDF5_file["wParamsNum"])

    # Create a dictionary with extracted metadata
    metadata_dict = {
        "filename": HDF5_file.filename,  # Name of the HDF5 file
        "exp_date": date,  # Date of the experiment
        "exp_time": time,  # Time of the experiment
        "objectiveXYZ": objectiveXYZ,  # Objective XYZ coordinates
    }

    return metadata_dict

def get_experiment_datetime(wParamsStr_arr):
    """
    Extract date and time of the experiment from the HDF5 file.

    Args:
        wParamsStr_arr (h5py.string): HDF5 string array containing the experiment date and time.

    Returns:
        tuple: A tuple containing the experiment date and time.
            - date (datetime.date): The date of the experiment.
            - time (datetime.time): The time of the experiment.
    """
    # Extract date and time from HDF5 string array
    date = wParamsStr_arr[4].decode("utf-8")  # Extract date string
    time = wParamsStr_arr[5].decode("utf-8")  # Extract time string

    # Split date and time strings into components and convert to integers
    date_components = np.array(date.split('-')).astype(int)
    time_components = np.array(time.split('-')).astype(int)

    # Create datetime.date and datetime.time objects from components
    date = datetime.date(date_components[0], date_components[1], date_components[2])
    time = datetime.time(time_components[0], time_components[1], time_components[2])

    # Return extracted date and time as a tuple
    return date, time

def get_rel_objective_XYZ(wParamsNum_arr):
    """Get xyz from wParamsNum"""
    wParamsNum_All = list(wParamsNum_arr)

    """
    Need to do this such that centering is done independently 
    for each plane in a series of files (maybe filter based on filename or smth).
    In this way, the objective position will be the offset in position from first 
    recording in any given series (but only within, never between experiments)

    Would it make sense to do this based on FishID maybe? Since new fish requires new mount and new location 
    """

    wParamsNum_All_XYZ = wParamsNum_All[26:29] # 26, 27, and 28 (X, Y, Z)
    X = wParamsNum_All_XYZ[0]
    Y = wParamsNum_All_XYZ[2]
    Z = wParamsNum_All_XYZ[1]
    return X, Y, Z

def get_regions(Exp: object) -> str:
    """
    Function to return the region of the Exp object based on the filename.

    Args:
        Exp (object): Exp object containing the filename.

    Returns:
        str: Region of the Exp object. Options are 'temporal', 'nasal', 'dorsal', 'strike', 'ventral'.
            Returns None if no region is found.
    """
    regions = ['temporal', 'nasal', 'dorsal', 'strike', 'ventral']
    
    # Iterate over the regions and return the region if it is in the filename
    for reg in regions:
        if reg in Exp.filename:
            return reg
    
    # If no region is found, return None
    return None
        
def get_pooled_IPL_positions(dir: str):
    """
    Get the pooled IPL positions from all h5 files in a directory.

    Args:
        dir (str): Path to the directory containing the h5 files.

    Returns:
        numpy.ndarray: Array of IPL positions from all h5 files.
    """
    # Get the list of h5 files in the directory
    files = get_h5_files(dir)
    regions = []
    # Iterate over the h5 files and append the IPL positions to the array
    for i, fpath in tqdm(enumerate(files)):        
        # Load the h5 file and get the IPL positions
        Exp = load_from_hdf5(fpath)        
        # Append the IPL positions to the array
        if i == 0:
            IPL_positions = Exp.Positions
        else:
            IPL_positions = np.append(IPL_positions, Exp.Positions, axis=0)
        
        for _ in Exp.Positions:
            regions.append(get_regions(Exp))

    # extend dimensions from (N, ) to (N, 1)
    IPL_positions = IPL_positions[:, np.newaxis]
    # Print the final shape of pooled traces
    print('final shape of pooled IPL positions:', IPL_positions.shape[0], 'rois')
    
    return IPL_positions, regions

from scipy.io import loadmat

def get_template(file, temp_type='3s'):
    """Load templates from .mat file as numpy array.

    Args:
        file (str): Path to .mat file
        temp_type (str, optional): Choose between '3s' and '2s'.
                                    Defaults to '3s'.

    Raises:
        ValueError: If temp_type is not '3s' or '2s'

    Returns:
        np.array: The loaded templates

    This function loads templates from a .mat file based on the
    provided file path and the template type. The template type
    should be either '3s' or '2s'. If a different template type is
    provided, a ValueError is raised. The function returns the
    loaded templates as a numpy array.
    """


    # Check if the provided template type is valid
    if temp_type == '3s':
        templates = loadmat(file)['templates_3s']     # Load the .mat file using scipy.io.loadmat
    elif temp_type == '2s':
        templates = loadmat(file)['templates_2s']    # Load the .mat file using scipy.io.loadmat
    else:
        raise ValueError('temp_type should be 3s or 2s')

    return templates


def get_interpolated_templates(templates: np.array, num_of_ds_ticks: int) -> np.array:
    """Interpolate templates to a specified number of ticks.
    
    Args:
        templates (np.array): The input templates to be interpolated.
        num_of_ds_ticks (int): The desired number of ticks in the interpolated templates.
        
    Returns:
        np.array: The interpolated templates.
    """
    # Get the number of ticks in the input templates and create equally spaced ticks
    num_of_ticks = templates.shape[0]
    ticks = np.linspace(0, num_of_ticks, num_of_ticks)
    
    # Create an interpolation function using the input templates
    f = interp.interp1d(ticks, templates, axis=0, fill_value='extrapolate')
    
    # Create equally spaced ticks for the interpolated templates
    ticks_interp = np.linspace(0, num_of_ticks, num_of_ds_ticks)
    
    # Interpolate the templates to the new ticks
    new_templates = f(ticks_interp)
    
    # Return the interpolated templates
    return new_templates

def get_downsampled_data(pooled_traces: np.array, num_of_templates_ticks: int) -> np.array:
    """Downsamples the input data to the desired number of ticks.
    
    Args:
        pooled_traces (np.array): The input data to be downsampled.
        num_of_templates_ticks (int): The desired number of ticks in the downsampled data.
        
    Returns:
        np.array: The downsampled data.
    """
    # Calculate the downsampling factor to achieve the desired number of ticks
    sampling_f = int(np.round((pooled_traces.shape[1] / num_of_templates_ticks)))
    
    # Downsample the data using the decimate function from scipy.signal
    ds_traces = sig.decimate(pooled_traces, sampling_f, n=8, axis=1)
    
    # Return the downsampled data
    return ds_traces

def get_closest_wavelength(df, wv):
	"""
	Find the index of the closest wavelength to the given wavelength in the dataframe.

	Parameters
	----------
	df : pandas.DataFrame
		A dataframe with a single column of wavelengths
	wv : float
		The wavelength to find the closest match to

	Returns
	-------
	int
		The index of the closest wavelength
	"""
	wvs = df.to_numpy()
	# Calculate the difference between the given wavelength and all the wavelengths in the dataframe
	diff = wvs - wv
	# Find the index of the minimum difference
	idx = (np.abs(diff)).argmin()
	return idx


def get_interpolated_cone_tunings(tunings: np.array, N_ticks_extra: int) -> np.array:
    """Interpolate cone tunings to a specified number of ticks.
    
    Args:
        tunings (np.array): The input tunings to be interpolated.
        N_ticks_extra (int): The desired number of ticks in the interpolated tunings.
        
    Returns:
        np.array: The interpolated tunings.
    """
    # Get the number of ticks in the input tunings and create equally spaced ticks
    n = tunings.shape[0]
    ticks = np.linspace(tunings['wavelength'].iloc[0], tunings['wavelength'].iloc[-1], n)

    # Create an interpolation function using the input tunings
    cols = list(tunings.columns) # list of column names
    f = interp.interp1d(ticks.T, tunings[cols[1:]].to_numpy(), axis=0, fill_value='extrapolate')

    # # Create equally spaced ticks for the interpolated tunings
    ticks2 = np.linspace(tunings['wavelength'].iloc[0], tunings['wavelength'].iloc[-1], N_ticks_extra)

    # # Interpolate the templates to the new ticks
    new_templates = f(ticks2)
    np_to_df = np.concatenate((ticks2[:, np.newaxis], new_templates), axis=1)
    # # # save to a data frame
    new_tunings = pd.DataFrame(np_to_df, columns=cols)

    # Return the interpolated templates
    return new_tunings

def get_cone_tunings(path_to_cone_data, N_ticks_extra):
    """
    Get interpolated cone tunings from the specified path to cone data.
    
    Args:
        path_to_cone_data (str): The path to the cone data file.
        N_ticks_extra (int): The desired number of ticks in the interpolated tunings.
        
    Returns:
        pd.DataFrame: The interpolated cone tunings.
    """
    # Read cone data from the specified path
    cone_tuning = pd.read_csv(path_to_cone_data)
    
    # Rename the column for wavelength
    cone_tuning = cone_tuning.rename(columns={'LED_wavelength': 'wavelength'})
    
    # Get the column names of the cone tuning data
    cone_tuning_columns = list(cone_tuning.columns)
    
    # Interpolate cone tunings to get extra ticks
    cone_tuning_extra = get_interpolated_cone_tunings(cone_tuning, N_ticks_extra=N_ticks_extra)
    
    # Invert the values of cone tunings for analysis
    cone_tuning_extra[cone_tuning_columns[1:]] = -1 * cone_tuning_extra[cone_tuning_columns[1:]]
    
    return cone_tuning_extra, cone_tuning_columns



# from: https://github.com/berenslab/cone_colour_tuning/blob/main/log_opsins_Fig1f_S4/opsin_log_transformation.ipynb
def optimizing_fun(params, x, data, return_fit=False):
    """
    Applies linear transformation on log(x+c) and normalizes to have max 1

    Parameters
    ----------
    params : tuple of floats
        (a, b, c_log) parameters of the linear transformation
    x : array-like
        input values
    data : array-like
        target values
    return_fit : bool, optional
        if True, returns the fit values in addition to the mean squared error

    Returns
    -------
    mse : float
        mean squared error between the fit and the data
    fit : array-like, optional
        fit values
    """
    a,b,c_log=params
    # scale the input values
    fit = a*np.log(x+c_log)+b
    # normalize to have max 1
    fit = fit/np.max(fit)
    # calculate the mean squared error between the fit and the data
    mse = np.mean((fit-data)**2)
    if return_fit:
        return fit, mse
    else:
        return mse

# from: https://github.com/berenslab/cone_colour_tuning/blob/main/log_opsins_Fig1f_S4/opsin_log_transformation.ipynb
def get_opsin_data():
    """
    Read opsin data from an excel file.
    
    Returns
    -------
    opsins : list
        List of opsins: ["R", "G", "B", "U"]
    experimental_wvs : pandas Series
        Indices of experimental wavelengths between min and max LED values
    opsin_df : pandas DataFrame
        DataFrame containing opsin data
    """
    fpath_expinfo = r'../experiment_info/zf_leds_for_analysis.json'
    led_nms = load_json(fpath_expinfo)
    
    led_nms_num = np.array([int(led) for led in led_nms])  # numerical values of the LED wavelengths [nm]
    min_led = np.min(led_nms_num)
    max_led = np.max(led_nms_num)
    
    filepath = r"..\experiment_info\model\LED_opsin_data.xlsx"
    opsin_df = pd.read_excel(filepath, header=0)  # [::-1]
    opsins = ["R", "G", "B", "U"]
    experimental_wvs = opsin_df['wavelength'].between(min_led, max_led)  # indices
    
    return opsins, experimental_wvs, opsin_df

# from: https://github.com/berenslab/cone_colour_tuning/blob/main/log_opsins_Fig1f_S4/opsin_log_transformation.ipynb
def get_log_opsin_data():
    """
    Read log opsin tuning curves and LED data for all cones.

    Returns
    -------
    hc_block_df : pandas DataFrame
        DataFrame containing the log opsin tuning curves for all cones.
    opsins : list
        List of opsins: ["R", "G", "B", "U"]
    experimental_wvs : pandas Series
        Indices of experimental wavelengths between min and max LED values.
    led_df : pandas DataFrame
        DataFrame containing the LED data.
    opsin_df : pandas DataFrame
        DataFrame containing opsin data.
    """
    # Get the opsin tuning curves (from Takeshi's paper)
    opsins, experimental_wvs, opsin_df = get_opsin_data()

    """
    Read LED data from an excel file.
    """
    # Define the file path
    filepath = r"..\experiment_info\model\LED wavelength.xlsx"
    # Read the excel file
    led_df = pd.read_excel(filepath, header=0)  # [::-1]

    """
    Read hc_block data for all cones from excel files.
    """
    # Define the file paths for all cones
    filepaths = [r"..\experiment_info\model\%s-Cone recordings - HCblock.xlsx" % opsin for opsin in opsins]
    # Check if the files exist
    for path in filepaths:
        if not os.path.exists(path):
            raise ValueError("File not found: %s" % path)
    # Read the excel files
    opsins_data = [pd.read_excel(path, header=0) for path in filepaths]
    # Merge the data frames
    hc_block_df = pd.concat(opsins_data, ignore_index=True)
    
    return hc_block_df, opsins, experimental_wvs, led_df, opsin_df

# from: https://github.com/berenslab/cone_colour_tuning/blob/main/log_opsins_Fig1f_S4/opsin_log_transformation.ipynb
def get_log_opsin_tunings():
    """
    Compute the log opsin tuning curves for all cone types by interpolating the
    mean control traces to the opsin wavelengths and then minimizing the mean
    squared error between the interpolated data and the opsin data.
    
    Returns
    -------
    log_opsin_tunings : pandas DataFrame
        DataFrame containing the log opsin tuning curves for all cone types
    """
    hc_block_df, opsins, experimental_wvs, led_df, opsin_df = get_log_opsin_data()
    
    res_all = []
    fit_all = []
    mse_all = []
    data = []

    for cone_type in opsins:
        # Compute mean control traces and interpolate to opsin wavelength
        wavelength = led_df[:13]['Wavelength'].values[::-1]
        data = hc_block_df[hc_block_df['cone_type']==cone_type].loc[:, hc_block_df.columns != 'cone_type'].mean(axis=0).values[::-1]
        data = data/np.max(data)
        
        opsin = opsin_df[(opsin_df['wavelength']>=360)&(opsin_df['wavelength']<=655) ][cone_type].values
        
        wavelength_interp = np.arange(360,656)
    
        data_interp = interp.interpolate.interp1d(wavelength,data)(wavelength_interp)
        
        # Minimize mean squared error
        res = optimize.minimize(optimizing_fun, x0=[0.5,2,0.1], 
                                    args=(opsin, data_interp),
                                bounds=((0,10),(0,10),(0,10)))
        
        # Extract results
        fit, mse = optimizing_fun(res.x,opsin, data_interp, return_fit=True )
        res_all.append(res)
        fit_all.append(fit)
        mse_all.append(mse)

    """
    Put into dataframe
    """
    log_opsin_tunings = pd.DataFrame(wavelength_interp, columns=['wavelength'])
    for i, cone_type in enumerate(opsins):

        log_opsin_tunings[cone_type] =  fit_all[i]
        
    return log_opsin_tunings, opsins


def get_leds_for_analysis() -> list:
    # Opening JSON file
    f = open(r'..\experiment_info\zf_leds_for_analysis.json')
    # make a list of conditions = subdirectories containing h5 files with data of a given condition
    leds = json.load(f)
    # Closing file
    f.close()
    return leds

def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength > 750:
        wavelength = 750.
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B, A)

def get_list_of_colors(nLEDs: int, red_first: bool) -> list:
    """
    Args:
        nLEDs (int): number of LEDs in the experiment

    Returns:
        list: list of colors used for ploting/visualization purposes - to show the LED ON events
    """
    leds = get_leds_for_analysis()
    wl = np.flipud(np.array([int(led) for led in leds]))
    clim = (min(wl), max(wl))
    norm = plt.Normalize(*clim)
    # wl = np.arange(clim[0], clim[1] + 1, 2)
    if red_first:
        cmap_list = np.flipud(np.array([wavelength_to_rgb(w) for w in wl]))
    else:
        cmap_list = np.array([wavelength_to_rgb(w) for w in wl])
    
    # spectralmap = LinearSegmentedColormap.from_list("nipy_spectral", colorlist)
    return cmap_list

def get_layered_weights(ws, step_size: int):

    results = []
    std = []
    num_cells = []
    interval = []
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in list(range(0, 100)[::step_size]): # range: 0-100 % of IPL
        start = i
        stop = i + step_size
        interval.append([start, stop])
        to_numpy = ws[ws["IPL"].between(start, stop)].drop(["IPL", 'condition'], axis = 1).to_numpy()
        nan_to_fill = np.full([to_numpy.shape[1]], np.nan)
        if to_numpy.shape[0] == 0:
            results.append(nan_to_fill)
            std.append(nan_to_fill)
            num_cells.append(to_numpy.shape[0])
        else:
            # to_numpy = scaler.fit_transform(to_numpy)
            results.append(np.mean(to_numpy, axis = 0))
            std.append(np.std(to_numpy, axis = 0))
            num_cells.append(to_numpy.shape[0])

    layered_ws_means = np.array(results)
    layered_ws_std = np.array(std)
    layered_num_cells = np.array(num_cells)
    return layered_ws_means, layered_ws_std, layered_num_cells, interval

def get_df_with_ipl_and_regions(ipl_positions: np.ndarray, regions: list, condition: str):
    """
    Create a pandas DataFrame with IPL positions and regions.

    Args:
        ipl_positions (np.ndarray): Array of IPL positions.
        regions (list): List of regions.
        condition (str): Condition of the data.

    Returns:
        pandas.DataFrame: DataFrame with IPL positions, regions, and condition.
    """
    # Create a DataFrame with IPL positions
    df = pd.DataFrame(data=ipl_positions, columns=['IPL'])
    # Add regions column to the DataFrame
    df['region'] = regions
    # Add condition column to the DataFrame
    df['condition'] = condition
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    return df
