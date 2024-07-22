import numpy as np
import numpy.ma as ma

def get_mask(traces, quality_criterion_all_traces, ipl_positions: np.array = None, qc: float = 0.35, std_qc: float=1):
    # Create a mask based on the quality criterion
    mask_std = np.max(np.abs(traces[:, :]), axis=1) < std_qc
    mask_std = mask_std[:, np.newaxis] # add a new axis to match the dimensions of other masks
    
    mask = quality_criterion_all_traces < qc    
    if ipl_positions is not None:
        # Create a mask for -1 positions - it means they are out of the IPL ROI
        mask_ipl = (ipl_positions == -1)    
        # Combine the masks using logical OR
        mask_or = mask | mask_ipl | mask_std
        return mask_or
    else:
        # Combine the masks using logical OR
        return mask | mask_std

def get_masked_traces(traces: np.array, 
                      quality_criterion_all_traces: np.array, 
                      qc: float,
                      ipl_positions: np.array = None, one_dim=False, std_qc: float = 1) -> np.ma.MaskedArray:
    """Masks the input traces array based on the quality criterion.
    
    Args:
        traces (np.array): The input traces array.
        quality_criterion_all_traces (np.array): The quality criterion array.
        qc (float, optional): The quality criterion threshold. Defaults to 0.35.
        
    Returns:
        np.ma.MaskedArray: The masked traces array.
        
    Raises:
        ValueError: If the input traces array is neither 2 nor 3 dimensional.
    """
    
    # Create a mask based on the quality criterion
    mask = get_mask(traces=traces, quality_criterion_all_traces=quality_criterion_all_traces, ipl_positions=ipl_positions,
                    qc=qc, std_qc=std_qc)
    
    # Broadcast the mask to the shape of the traces array
    if traces.ndim == 3:
        # For 3D arrays, add a new axis to broadcast the mask
        mask = np.broadcast_to(mask[:, np.newaxis], 
                                (traces.shape[0], traces.shape[1], 
                                 traces.shape[2])).copy()
        traces_masked = ma.array(traces, mask=mask)
    elif traces.ndim == 2:
        # For 2D arrays, broadcast the mask directly
        mask = np.broadcast_to(mask[:], (traces.shape[0], traces.shape[1])).copy()
        traces_masked = ma.array(traces, mask=mask)
    elif traces.ndim == 1 and one_dim == True:
        # For 1D arrays, broadcast the mask directly
        traces_masked = ma.array(traces, mask=mask)
    else:
        # Raise an error if the input traces array is neither 2 nor 3 dimensional
        raise ValueError('traces must be 2 or 3 dimensional')
    
    # Create a masked array using the mask
    
    
    return traces_masked

def get_masked_IPL_positions(ipl_positions: np.ndarray, traces: np.ndarray, quality_criterion_all_traces: np.ndarray, qc: float = 0.35, std_qc: float = 1):
    """
    Get masked IPL positions based on a quality criterion and -1 IPL positions.

    Args:
        ipl_positions (np.ndarray): Array of IPL positions.
        quality_criterion_all_traces (np.ndarray): Array of quality criterion values.
        qc (float): Quality criterion threshold.

    Returns:
        numpy.ma.MaskedArray: Masked array of IPL positions where quality criterion is less than the threshold
        or IPL position is -1.
    """
    # Create a mask based on the quality criterion and IPL positions
    mask = get_mask(traces=traces, quality_criterion_all_traces=quality_criterion_all_traces, ipl_positions=ipl_positions,
                    qc=qc, std_qc=std_qc)
    if isinstance(mask, np.ma.core.MaskedArray):
        mask = mask.mask
    # Create a masked array using the mask
    positions_masked = ma.masked_array(ipl_positions, mask=mask)
    
    return positions_masked

def get_masked_recorded_regions(regions: np.ndarray, ipl_positions: np.ndarray, traces: np.ndarray, quality_criterion_all_traces: np.ndarray, qc: float = 0.35, std_qc: float = 1):
    """
    Get masked IPL positions based on a quality criterion and -1 IPL positions.

    Args:
        ipl_positions (np.ndarray): Array of IPL positions.
        quality_criterion_all_traces (np.ndarray): Array of quality criterion values.
        qc (float): Quality criterion threshold.

    Returns:
        numpy.ma.MaskedArray: Masked array of IPL positions where quality criterion is less than the threshold
        or IPL position is -1.
    """
    # Create a mask based on the quality criterion and IPL positions
    mask = get_mask(traces=traces, quality_criterion_all_traces=quality_criterion_all_traces, ipl_positions=ipl_positions,
                    qc=qc, std_qc=std_qc)
    if isinstance(mask, np.ma.core.MaskedArray):
        mask = mask.mask
    # Create a masked array using the mask
    regions_masked = ma.masked_array(regions, mask=mask)
    
    return regions_masked
