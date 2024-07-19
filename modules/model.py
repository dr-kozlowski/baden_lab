import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy.interpolate as interp
import scipy.optimize as optimize

# Python3 program to find Closest number in a list
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
    
    return cone_tuning_extra
