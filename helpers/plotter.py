import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def compare_map(x_test,y_test,y_predict):
    if x_test.shape[1]>1:
        print("Since x_test is multivariate, only using the first dimension to plot.")
    if(len(x_test)<1000):
        plot = plt.scatter
    else:
        plot = plt.plot
    plot(x_test[:,0],y_test[:,0],label="Data")
    plot(x_test[:,0],y_predict[:,0],label="Forecast")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def compare_output_timeseries(y_test,y_pred,y_error=None,display_rmse=False):
    if y_test.shape[1]>1:
        print("Since y_test is multivariate, only using the first dimension to plot.")
    plot = plt.plot
    t = range(len(y_test))
    plot(t,y_test[:,0],label="Data")
    plot(t,y_pred[:,0],label="Forecast",color='orangered')
    if isinstance(y_error,np.ndarray):
        plt.fill_between(t,y_pred.flatten()-y_error.flatten(),y_pred.flatten()+y_error.flatten(),color='orangered',alpha=0.5)
    plt.xlabel("t (index)")
    plt.ylabel("y")
    plt.legend()
    if display_rmse:
        plt.title("RMSError = "+str('{:03.2f}'.format(np.sqrt(mean_squared_error(y_test,y_pred)))))
    plt.show()
    
