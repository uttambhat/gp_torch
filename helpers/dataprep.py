import numpy as np

def normalize(time_series_data):
    """
    Normalizes the time series data to have zero mean and unit standard deviation
    
    Returns:
    Normalized time series, time series mean (by variable), standard deviation (by variable)
    """
    mean = np.mean(time_series_data,axis=0)
    stdev = np.sqrt(np.var(time_series_data,axis=0))
    time_series_data_normalized = ((time_series_data-mean)/stdev)
    return time_series_data_normalized,mean,stdev

