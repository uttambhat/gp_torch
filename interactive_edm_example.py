#===================================================
# Fitting gpr_known_noise via interactive plot
# CONSIDER CONVERTING THIS TO A JUPYTER NOTEBOOK!
#===================================================

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from gp_torch.gpr_known_noise import *
import matplotlib.pyplot as plt

# test x values
x_test = torch.arange(-2.,2.,0.01).reshape(-1,1)
x_grid = torch.arange(-2.,2.,0.1)

# Define model without training data (training data added interactively through the plot)
model = gaussian_process_regressor(prior='ard') #Alternate prior=None
x_count = 0
x_train_m1 = 0
num_forecast_steps = 2

# Interactive plot. User can add high-fidelity points via left-clicks and low-fidelity points via right-clicks
fig = plt.figure(figsize=(20,10))
ax = {}
ax[0] = fig.add_subplot(121)
ax[0].set_xlim(x_test.min(),x_test.max())
ax[0].set_ylim(x_test.min(),x_test.max())
y_pred,y_error = model.predict(x_test,return_std=True)

ax[1] = fig.add_subplot(122)
ax[1].set_xlim(x_grid.min(),x_grid.max())
ax[1].set_ylim(x_grid.min(),x_grid.max())
ax[1].plot(x_test.detach().numpy(),y_pred.detach().numpy())
ax[1].fill_between(x_test.detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_error.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_error.detach().numpy().flatten(),color='orangered',alpha=0.4)
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
    ax[0].scatter([x_grid[x_count]], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
    ax[0].errorbar([x_grid[x_count]], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    if x_count == 0:
        x_train_m1 = y
        del ax[1].collections[-1]
    elif x_count == 1:
        model.x_train = torch.cat((model.x_train,torch.tensor([[x_train_m1]]))) # x_train takes previous x_train[-1] #EDM
        model.y_train = torch.cat((model.y_train,torch.tensor([[y]]))) 
        model.noise = torch.cat((model.noise,torch.tensor([noise])))
        model.n_train += 1
        del ax[1].collections[-1]
        ax[1].scatter([model.x_train[-1,0]], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
        ax[1].errorbar([model.x_train[-1,0]], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    elif x_count > 1:
        model.x_train = torch.cat((model.x_train,model.y_train[-1].reshape(-1,1))) # x_train takes previous x_train[-1] #EDM
        model.y_train = torch.cat((model.y_train,torch.tensor([[y]]))) 
        model.noise = torch.cat((model.noise,torch.tensor([noise])))
        model.optimize(iterations=1000)
        del ax[1].collections[-1]
        ax[1].scatter([model.x_train[-1,0]], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
        ax[1].errorbar([model.x_train[-1,0]], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    print(model.phi.detach(),model.tau.detach(),model.noise)
    if x_count>0 and x_count+num_forecast_steps<len(x_grid):
        x_next_pred,x_next_err = model.predict(model.y_train[-1].reshape(-1,1))
        ax[0].scatter([x_grid[x_count+1]], [x_next_pred.detach()[0,0]], c='orangered', s=150, alpha=1./(1.+10.*(x_next_err.detach()[0,0].numpy()**2)))
        ax[0].errorbar([x_grid[x_count+1]], [x_next_pred.detach()[0,0]], yerr=x_next_err.detach()[0,0], c='orangered',alpha=1./(1.+10.*(x_next_err.detach()[0,0].numpy()**2)))
        for i in range(1,num_forecast_steps):
            x_next_pred,x_next_err = model.predict(x_next_pred.detach())
            ax[0].scatter([x_grid[x_count+i+1]], [x_next_pred.detach()[0,0]], c='orangered', s=150, alpha=1./(1.+((i+1.)**2)*10.*(x_next_err.detach()[0,0].numpy()**2)))
            ax[0].errorbar([x_grid[x_count+i+1]], [x_next_pred.detach()[0,0]], yerr=x_next_err.detach()[0,0], c='orangered',alpha=1./(1.+((i+1.)**2)*10.*(x_next_err.detach()[0,0].numpy()**2)))
    y_pred,y_error = model.predict(x_test,return_std=True)
    del ax[1].lines[:]
    ax[1].plot(x_test.detach().numpy(),y_pred.detach().numpy(),color='orangered')
    ax[1].fill_between(x_test.detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_error.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_error.detach().numpy().flatten(),color='orangered',alpha=0.5)
    plt.draw()
    x_count += 1

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.show()

