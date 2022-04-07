#===================================================
# Fitting gpr_known_noise via interactive plot
# CONSIDER CONVERTING THIS TO A JUPYTER NOTEBOOK!
#===================================================

import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from gp_torch.gpr_known_noise import *
from tests.fnn import fnn
import matplotlib.pyplot as plt

# test x values
x_test = torch.arange(-2.,2.,0.01).reshape(-1,1)

# Define model without training data (training data added interactively through the plot)
model_gp = gaussian_process_regressor(prior='ard') #Alternate prior=None
model_nn = fnn()
model_nn.initialize(np.zeros((0,1)), np.zeros((0,1)), hidden_units=100,activation='relu')
initial_theta=np.array(list(model_nn.parameters.values()))

# Interactive plot. User can add high-fidelity points via left-clicks and low-fidelity points via right-clicks
plt.rcParams.update({'font.size':16})
line_thickness = 3
fig = plt.figure(figsize=(20,10))
ax = {}
ax[0] = fig.add_subplot(121)
ax[0].set_title("GP")
ax[0].set_xlim(x_test.min(),x_test.max())
ax[0].set_ylim(x_test.min(),x_test.max())
y_pred,y_error = model_gp.predict(x_test,return_std=True)
ax[0].plot(x_test.detach().numpy(),y_pred.detach().numpy(),color='orangered',lw=line_thickness)
ax[0].fill_between(x_test.detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_error.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_error.detach().numpy().flatten(),color='orangered',alpha=0.4)

ax[1] = fig.add_subplot(122)
ax[1].set_title("NN")
ax[1].set_xlim(x_test.min(),x_test.max())
ax[1].set_ylim(x_test.min(),x_test.max())
y_pred = model_nn.predict(x_test.detach().numpy())
ax[1].plot(x_test,y_pred,color='orangered',lw=line_thickness)

def onclick(event):
    x, y = event.xdata, event.ydata
    if event.button == 1:
        noise = 0.1
    elif event.button == 3:
        noise = 0.5
    print(f'{x:0.1f},{y:0.1f}')
    global model_gp, model_nn
    
    model_gp.x_train = torch.cat((model_gp.x_train,torch.tensor([[x]])))
    model_gp.y_train = torch.cat((model_gp.y_train,torch.tensor([[y]])))
    model_gp.noise = torch.cat((model_gp.noise,torch.tensor([noise])))
    model_gp.optimize(iterations=1000)
    print(model_gp.phi.detach(),model_gp.tau.detach(),model_gp.noise)
    del ax[0].lines[-1]
    del ax[0].collections[-1]
    ax[0].scatter([x], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
    ax[0].errorbar([x], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    y_pred,y_error = model_gp.predict(x_test,return_std=True)
    ax[0].plot(x_test.detach().numpy(),y_pred.detach().numpy(),color='orangered',lw=line_thickness)
    ax[0].fill_between(x_test.detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_error.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_error.detach().numpy().flatten(),color='orangered',alpha=0.5)
    
    model_nn.x_train = np.concatenate((model_nn.x_train,np.array([x]).reshape(-1,1)),axis=0)
    model_nn.y_train = np.concatenate((model_nn.y_train,np.array([y]).reshape(-1,1)),axis=0)
    initial_theta=np.array(list(model_nn.parameters.values()))
    (loss_vector,loss_val_vector)=model_nn.train(model_nn.x_train,model_nn.y_train,initial_theta,train_validation_split=1.,epochs=1000,restore_best_theta=True)
    del ax[1].lines[-1]
    ax[1].scatter([x], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
    ax[1].errorbar([x], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    y_pred = model_nn.predict(x_test.detach().numpy())
    ax[1].plot(x_test.detach().numpy(),y_pred,color='orangered',lw=line_thickness)
    
    plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.show()

