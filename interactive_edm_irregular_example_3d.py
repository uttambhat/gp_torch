#===================================================
# Fitting gpr_known_noise via interactive plot
# CONSIDER CONVERTING THIS TO A JUPYTER NOTEBOOK!
#===================================================
# TO BE CLEANED!!!!!!!!!!!!!
#===================================================

import torch
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)
from gp_torch.gpr_known_noise_for_irregular_erroneous import *
import matplotlib.pyplot as plt

# test x values
x_test = torch.arange(-2.,2.,0.01).reshape(-1,1)
x_test = torch.cat((x_test,torch.zeros(x_test.shape)),axis=1)
x_grid = torch.arange(-2.,2.,0.1)
x_train = torch.arange(-2.,2.,0.1).reshape(-1,1)
y_train = x_train.clone()
x_train = torch.cat((x_train,torch.zeros(x_train.shape)),axis=1)

X = np.arange(-2, 2, 0.25)
Y = np.arange(0., 2, 0.25)
X, Y = np.meshgrid(X, Y)
x_grid_test = torch.cat((torch.from_numpy(X.reshape(-1,1)),torch.from_numpy(Y.reshape(-1,1))),axis=1)
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Define model without training data (training data added interactively through the plot)
model = gaussian_process_regressor(x_train=x_train,y_train=y_train,prior='ard') #Alternate prior=None
x_count = 0
x_train_m1 = 0
num_forecast_steps = 2

# Interactive plot. User can add high-fidelity points via left-clicks and low-fidelity points via right-clicks
fig = plt.figure(figsize=(20,10))
ax = {}
ax[0] = fig.add_subplot(121)
ax[0].set_xlim(x_test[:,0].min(),x_test[:,0].max())
ax[0].set_ylim(x_test[:,0].min(),x_test[:,0].max())
y_pred,y_error = model.predict(x_grid_test,return_std=True)

ax[1] = fig.add_subplot(122, projection='3d')
ax[1].set_xlim(x_grid.min(),x_grid.max())
ax[1].set_ylim(x_grid.min(),x_grid.max())
#ax[1].scatter(x_train[:,0].detach(), y_train[:,0].detach(), c='b', s=20)
#ax[1].plot(x_test[:,0].detach().numpy(),y_pred[:,0].detach().numpy())
#ax[1].fill_between(x_test[:,0].detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_error.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_error.detach().numpy().flatten(),color='orangered',alpha=0.4)
ax[1].plot_surface(X,Y,y_pred.reshape(X.shape).detach().numpy(), color = 'orangered')
ax[1].set_xlabel("x_-1")
ax[1].set_ylabel("dx")
ax[1].set_zlabel("x_pred")
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
        #del ax[1].collections[-1]
    elif x_count == 1:
        model.x_train = torch.cat((model.x_train,torch.tensor([[x_train_m1,x_list[-1]-x_list[-2]]]))) # x_train takes previous x_train[-1] #EDM
        model.y_train = torch.cat((model.y_train,torch.tensor([[y]]))) 
        model.noise = torch.cat((model.noise,torch.tensor([noise])))
        model.n_train += 1
        #del ax[1].collections[-1]
        #ax[1].scatter([model.x_train[-1,0]], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
        #ax[1].errorbar([model.x_train[-1,0]], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    elif x_count > 1:
        model.x_train = torch.cat((model.x_train,torch.cat((model.y_train[-1],torch.tensor([x_list[-1]-x_list[-2]]))).reshape(-1,2))) # x_train takes previous x_train[-1] #EDM
        model.y_train = torch.cat((model.y_train,torch.tensor([[y]]))) 
        model.noise = torch.cat((model.noise,torch.tensor([noise])))
        model.optimize(iterations=1000)
        #del ax[1].collections[-1]
        #ax[1].scatter([model.x_train[-1,0]], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
        #ax[1].errorbar([model.x_train[-1,0]], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    print(model.phi.detach(),model.tau.detach(),model.noise)
    if x_count>0:
        x_right = torch.arange(x_list[-1],x_grid[-1],0.01).reshape(-1,1)
        x_right_test = torch.cat((torch.ones_like(x_right+x_list[-1])*model.y_train[-1,0],x_right-x_list[-1]),axis=1)
        x_next_pred,x_next_err = model.predict(x_right_test)
        del ax[0].lines[:]
        del ax[0].collections[-3]
        ax[0].plot(x_right.detach().numpy(),x_next_pred.detach().numpy(),color='orangered')
        ax[0].fill_between(x_right.detach().numpy().flatten(),x_next_pred.detach().numpy().flatten()-x_next_err.detach().numpy().flatten(),x_next_pred.detach().numpy().flatten()+x_next_err.detach().numpy().flatten(),color='orangered',alpha=0.5)
    y_pred,y_error = model.predict(x_grid_test,return_std=True)
    del ax[1].collections[:]
    ax[1].plot_surface(X,Y,y_pred.reshape(X.shape).detach().numpy(), color = 'orangered')
    plt.draw()
    x_count += 1

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.show()

