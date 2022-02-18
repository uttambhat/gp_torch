import numpy as np

def construct(time_series_data,x_columns,y_columns,number_of_lags,lag=1,forecast_steps_ahead=1,test_fraction=0.2):
    """
    Constructs Empirical Delay Modeling lag vectors from time series data
    
    Example usage:
    x_columns = [0,1] #independent variables
    y_columns = [0]     #dependent variables
    number_of_lags = [2,3] #number of time-lags for each independent variable
    lag = [1,2] #lag for each variable (or scalar for same lag across all variables)
    forecast_steps_ahead = 1 #number of time-steps ahead for forecasting
    test_fraction = 0.2 #fraction of data to be used to calculate out-of-sample error (can be set to zero to use all data for training)
    x_train,y_train,x_test,y_test = edm.simple.construct(data,x_columns,y_columns,number_of_lags,lag,forecast_steps_ahead,test_fraction)
    """
    if not len(x_columns)==len(number_of_lags)==len(lag):
        raise ValueError("Make sure the lengths of lists x_columsn, number_of_lags and lag matches")

    number_of_lags = np.array(number_of_lags)
    lag = np.array(lag)
    max_lag = np.max(number_of_lags*lag + forecast_steps_ahead)
    if max_lag>=len(time_series_data):
        raise ValueError("Lags too large for given length of time_series_data!")
    x,y = [],[]
    x_columns_expanded = np.concatenate(([np.ones(number_of_lags[i])*x_columns[i] for i in range(len(number_of_lags))]))
    lags_expanded = np.concatenate(([np.arange(1,number_of_lags[i]+1)*lag[i] for i in range(len(lag))]))
    for i in range(max_lag-forecast_steps_ahead,len(time_series_data)-forecast_steps_ahead):
        x_vector = time_series_data[i-lags_expanded,x_columns_expanded.astype(int)]
        x.append(x_vector)
        y_vector = time_series_data[i+forecast_steps_ahead,y_columns]
        y.append(y_vector)

    x = np.array(x)
    y = np.array(y)
    n = len(x)
    train_length = int((1.-test_fraction)*n)
    x_train,y_train = x[:train_length],y[:train_length]
    x_test,y_test = x[train_length:],y[train_length:]
    return x_train,y_train,x_test,y_test
    
