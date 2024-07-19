import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.signal as sig
import scipy.interpolate as interp
import scipy.optimize as optimize

def fit_model(traces: np.array, templates: np.array):
    """
    Fit the pooled traces to the templates and return the fitted responses and weights.

    Args:
        traces (np.array): The pooled traces to be fitted.
        templates (np.array): The input templates to fit the pooled traces to.

    Returns:
        fitted_responses (np.array): The fitted responses.
        weights (np.array): The weights.

    Raises:
        AssertionError: If the shapes of pooled_traces and templates do not match.
    """
    
    # Check if the shapes of pooled_traces and templates match
    try:
        assert traces.shape[1] == templates.shape[0], "pooled_traces.shape[1] != templates.shape[0]"
    except AssertionError as e:
        raise AssertionError(e)
    
    
    # Initialize arrays for the fitted responses and weights
    fitted_responses = np.zeros(traces.shape)
    weights = np.zeros((traces.shape[0], templates.shape[1]))
    
    # Fit the pooled traces to the templates and calculate the fitted responses and weights
    for id, roi in enumerate(traces):
        # Calculate the weights using linear least squares
        x = np.linalg.lstsq(templates, roi, rcond=None)[0]
        weights[id] = x.T
        # Calculate the fitted responses using the weights and the templates
        fitted_responses[id] = np.dot(templates, x)

    # Create masks for the fitted responses and weights based on the pooled traces mask
    if ma.is_masked(traces):
        # Broadcast the mask to the shape of the weights
        weights_mask = np.broadcast_to(traces.mask[:, 0, None], weights.shape)
        # Create a masked array for the fitted responses
        fitted_responses = ma.array(fitted_responses, mask=traces.mask)
        # Create a masked array for the weights
        weights = ma.array(weights, mask=weights_mask)
        # Return the fitted responses and weights as masked arrays
        return fitted_responses, weights
    else:
        # Return the fitted responses and weights as regular arrays
        return fitted_responses, weights

    