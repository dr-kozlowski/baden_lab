import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
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

def get_residuals(fits, traces):
    """
    Calculate the fitted traces and the experimental traces from the given fitted and pooled ON-OFF traces.
    Then, calculate the residuals by subtracting the experimental traces from the fitted traces.
    
    Args:
        fits (np.array): The fitted ON-OFF traces.
        traces (np.array): The pooled ON-OFF traces.
        nLEDs (int): The number of LEDs.
        
    Returns:
        residuals (np.array): The residuals.
    """    
    traces = traces.compressed().reshape(-1, traces.shape[1])
    fits = fits.compressed().reshape(-1, traces.shape[1])
    # Calculate the residuals by subtracting the experimental traces from the fitted traces
    residuals = fits - traces
    
    return residuals, traces, fits

def plot_residuals(exps, lstsq_fits, lstsq_resids, ticks_per_template, nLEDs, cmap_list, c, alpha, condition):
    N = ticks_per_template * nLEDs
    ticks = int(N / (nLEDs))
    aspect_ratio = 1920 / 1080
    height = 10
    fig, axs = plt.subplots(1, 3, figsize=(height*aspect_ratio, height))
    fontsize_tit = 22
    fontsize = 20
    ass = 'auto'

    im1 = axs[0].imshow(exps, interpolation='None', cmap='jet', aspect=ass)
    im2 = axs[1].imshow(lstsq_fits, interpolation='None', cmap='jet', aspect=ass)
    im3 = axs[2].imshow(lstsq_resids, interpolation='None', cmap='jet', aspect=ass)
    cbar1 = fig.colorbar(im1, ax=axs[0])
    cbar2 = fig.colorbar(im2, ax=axs[1])
    cbar3 = fig.colorbar(im3, ax=axs[2])
    cbar1.ax.tick_params(labelsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    axs[0].set_title('average traces', fontsize=fontsize_tit)
    axs[1].set_title('LSTSQ fits', fontsize=fontsize_tit)
    axs[2].set_title('LSTSQ residuals', fontsize=fontsize_tit)

    axs[1].set_xlabel('time points')

    for i in range(nLEDs):
        axs[0].axvline((i*ticks), ymin=-0.01, ymax=0.01, c = cmap_list[i], lw = 5)
        axs[1].axvline((i*ticks), ymin=-0.01, ymax=0.01, c = cmap_list[i], lw = 5)
        axs[2].axvline((i*ticks), ymin=-0.01, ymax=0.01,c = cmap_list[i], lw = 5)
    for i in range(2*nLEDs+1):
        axs[0].axvline((i*ticks/2), c = 'k', lw = 1)
        axs[1].axvline((i*ticks/2), c = 'k', lw = 1)
        axs[2].axvline((i*ticks/2), c = 'k', lw = 1)
    fig.suptitle('{}: residuals'.format(condition), fontsize=fontsize_tit+5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig('residuals_{}.png'.format(condition), dpi=600)
    fig.tight_layout()
    # return fig
    