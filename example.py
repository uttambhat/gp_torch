#=============================================================
# Simple example fitting a noisy sine wave using standard gpr
#=============================================================

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from gp_torch.gpr import *
import matplotlib.pyplot as plt

# Create training data. 2d tensors necessary
x_train = torch.arange(0.,1.,0.01).reshape(-1,1)
y_train = torch.sin(x_train*10.) + 0.1*torch.randn(x_train.shape)

# Define model and optimize parameters
model = gaussian_process_regressor(x_train,y_train,prior='ard') #Alternate prior=None
model.optimize()

# Predict y for x_train (you may define a separate x_test to perform out-of-sample prediction
y_pred = model.predict(x_train)
plt.scatter(x_train.detach().numpy(),y_train.detach().numpy())
plt.scatter(x_train.detach().numpy(),y_pred.detach().numpy())
plt.show()

