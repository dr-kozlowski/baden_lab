import numpy as np

def get_triggertimes_frame(Experiment: object, n: int = 3) -> np.array:
    """
    Function to get the trigger times in frames. Removes NaNs and slices
    the array to get the trigger times of the LEDs ON and OFF events.

    Args:
        Experiment (object): The experiment object.
        n (int, optional): Number of frames to skip between each event. Defaults to 3.

    Returns:
        np.array: The array of trigger times in frames, where each value is the frame of the event.
    """

    # Get the trigger times in frames
    triggertimes = Experiment.Triggertimes_Frame

    # Remove NaNs from the array
    triggertimes = triggertimes[~np.isnan(triggertimes)]

    # Slice the array to get the trigger times of the LEDs ON and OFF events
    triggertimes = triggertimes[0:-1:n]

    return triggertimes

def get_triggertimes_ms(Experiment: object, n: int = 3) -> np.array:
    """
    Small function to clean up the array of trigger times [ms] that corresponds to the LEDs ON and OFF events.

    Args:
        Experiment (object): The experiment object.
        n (int, optional): Number of frames to skip between each event. Defaults to 3.

    Returns:
        np.array: The array of trigger times [ms] where each value is the time in milliseconds when the event occurred.
    """
    # Get the trigger times in ms
    triggertimes = Experiment.Triggertimes

    # Remove NaNs from the array
    triggertimes = triggertimes[~np.isnan(triggertimes)]

    # Slice the array to get the trigger times of the LEDs ON and OFF events
    triggertimes = triggertimes[0:-1:n]

    return triggertimes

def get_triggerstamps(triggertimes: np.array, nLEDs: int) -> list:
    """
    Function to calculate the trigger stamps for each LED based on the trigger times.

    Args:
        triggertimes (np.array): trigger times [frame] that corresponds to the LEDs ON and OFF
        nLEDs (int): number of LEDs in the experiment

    Returns:
        trigger_stamps (list): list of lists containing all the x-axis frame stamps [x_start, x_end]
                               for each start and stop of LED ON events
    """
    # Initialize an empty list to hold the trigger stamps
    trigger_stamps = []

    # Iterate over each LED
    for led in range(nLEDs):
        # Create a list of lists containing the x-axis frame stamps [x_start, x_end]
        # for each start and stop of LED ON events
        xs_temp = [[triggertimes[i + 2*led], triggertimes[i+1 + 2*led]]
                   for i in range(1, len(triggertimes)-nLEDs, 2*nLEDs)]
        # Append the list to the trigger_stamps list
        trigger_stamps.append(xs_temp)
    
    return trigger_stamps


def timeshift_triggerstamps(trigger_stamps: list, t_shift: float) -> list:
    """_summary_

    Args:
        trigger_stamps (list): list to be time-shifted
        t_shift (float): value to time-shift with

    Returns:
        list: time-shifted trigger_stamps
    """
    return list(map(lambda x: x - t_shift, trigger_stamps))
