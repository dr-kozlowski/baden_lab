import numpy as np
import numpy.ma as ma

def get_mask(traces, quality_criterion_all_traces, qc, ipl_positions, std_qc=1):
    # Create a mask based on the quality criterion
    mask_std = np.abs(traces[:, :]) > std_qc
    mask = quality_criterion_all_traces < qc    
    if ipl_positions is not None:
        # Create a mask for -1 positions - it means they are out of the IPL ROI
        mask2 = (ipl_positions == -1)    
        # Combine the masks using logical OR
        mask_OR = mask | mask2 | mask_std
        return mask_OR
    else:
        # Combine the masks using logical OR
        return mask | mask_std

def get_masked_traces(traces: np.array, 
                      quality_criterion_all_traces: np.array, 
                      qc: float = 0.35,
                      IPL_positions: np.array = None, one_dim=False, std_qc: float = 1) -> np.ma.MaskedArray:
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
    mask = get_mask(traces, quality_criterion_all_traces,
                    qc, IPL_positions, std_qc=std_qc)
    
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