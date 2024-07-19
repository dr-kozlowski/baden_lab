import os
import json
from dataclasses import dataclass, field
import h5py
from tqdm.auto import tqdm
from dacite import from_dict
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import datetime
import scipy.signal as sig
import scipy.interpolate as interp
import pandas as pd

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