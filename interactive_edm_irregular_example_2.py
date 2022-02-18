#===================================================
# Fitting gpr_known_noise via interactive plot
# CONSIDER CONVERTING THIS TO A JUPYTER NOTEBOOK!
#===================================================
# TO BE CLEANED!!!!!!!!!!!!!
#===================================================

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from gp_torch.gpr_known_noise_for_irregular_erroneous import *
import matplotlib.pyplot as plt

# test x values
x_grid = torch.arange(-2.,2.,0.01)
x_train = torch.arange(-2.,2.,0.1).reshape(-1,1)
y_train = x_train.clone()
x_train = torch.cat((x_train,torch.zeros(x_train.shape)),axis=1)
n_initial = x_train.shape[0]

# Define model without training data (training data added interactively through the plot)
model = gaussian_process_regressor(x_train=x_train,y_train=y_train,prior='ard') #Alternate prior=None
x_count = 0
x_train_m1 = 0
num_forecast_steps = 2

# Interactive plot. User can add high-fidelity points via left-clicks and low-fidelity points via right-clicks
fig = plt.figure(figsize=(14,10))
ax = {}
ax[0] = fig.add_subplot(111)
ax[0].set_xlim(x_grid.min(),x_grid.max())
ax[0].set_ylim(x_grid.min(),x_grid.max())
x_list = []
def onclick(event):
    x, y = event.xdata, event.ydata
    if event.button == 1:
        noise = 0.1
    elif event.button == 3:
        noise = 0.5
    print(f'{x:0.2f},{y:0.2f}')
    global model
    global x_count
    global x_train_m1
    global x_list
    x_list.append(x)
    ax[0].scatter([x], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
    ax[0].errorbar([x], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    if x_count == 0:
        x_train_m1 = y
    elif x_count == 1:
        model.x_train = torch.cat((model.x_train,torch.tensor([[x_train_m1,x_list[-1]-x_list[-2]]]))) # x_train takes previous x_train[-1] #EDM
        model.y_train = torch.cat((model.y_train,torch.tensor([[y]]))) 
        model.noise = torch.cat((model.noise,torch.tensor([noise])))
        model.n_train += 1
    elif x_count > 1:
        model.x_train = torch.cat((model.x_train,torch.cat((model.y_train[-1],torch.tensor([x_list[-1]-x_list[-2]]))).reshape(-1,2))) # x_train takes previous x_train[-1] #EDM
        model.y_train = torch.cat((model.y_train,torch.tensor([[y]]))) 
        model.noise = torch.cat((model.noise,torch.tensor([noise])))
        model.optimize(iterations=1000)
    print(model.phi.detach(),model.tau.detach(),model.noise)
    if x_count>0:
        dists,indices=torch.cdist(x_grid.reshape(-1,1),torch.tensor(x_list[1:]).reshape(-1,1)).min(axis=1)
        x_test = torch.cat((model.y_train[n_initial+indices],dists.reshape(-1,1)),axis=1)
        y_pred,y_pred_err = model.predict(x_test)
        del ax[0].lines[:]
        del ax[0].collections[-3]
        ax[0].plot(x_grid.detach().numpy(),y_pred.detach().numpy(),color='orangered')
        ax[0].fill_between(x_grid.detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_pred_err.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_pred_err.detach().numpy().flatten(),color='orangered',alpha=0.5)
    plt.draw()
    x_count += 1

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.show()

