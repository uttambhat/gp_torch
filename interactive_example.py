#=============================================
# Fitting gpr_known_noise via interactive plot
#=============================================

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from gp_torch.gpr_known_noise import *
import matplotlib.pyplot as plt

# test x values
x_test = torch.arange(-1.,1.,0.01).reshape(-1,1)

# Define model without training data (training data added interactively through the plot)
model = gaussian_process_regressor(prior='ard') #Alternate prior=None

# Interactive plot. User can add high-fidelity points via left-clicks and low-fidelity points via right-clicks
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlim(x_test.min(),x_test.max())
ax.set_ylim(x_test.min(),x_test.max())
y_pred,y_error = model.predict(x_test,return_std=True)
ax.scatter(x_train.detach().numpy(),y_train.detach().numpy())
ax.plot(x_test.detach().numpy(),y_pred.detach().numpy())
plt.fill_between(x_test.detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_error.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_error.detach().numpy().flatten(),color='orangered',alpha=0.4)
def onclick(event):
    x, y = event.xdata, event.ydata
    if event.button == 1:
        noise = 0.1
    elif event.button == 3:
        noise = 0.5
    print(f'{x:0.1f},{y:0.1f}')
    global model
    model.x_train = torch.cat((model.x_train,torch.tensor([[x]])))
    model.y_train = torch.cat((model.y_train,torch.tensor([[y]])))
    model.noise = torch.cat((model.noise,torch.tensor([noise])))
    model.n_train += 1
    model.optimize(iterations=1000)
    print(model.phi,model.tau,model.noise)
    del ax.lines[-1]
    del ax.collections[-1]
    ax.scatter([x], [y], c='b', s=150, alpha=1./(1.+10.*(noise**2)))
    ax.errorbar([x], [y], yerr=noise, c='b',alpha=1./(1.+10.*(noise**2)))
    y_pred,y_error = model.predict(x_test,return_std=True)
    ax.plot(x_test.detach().numpy(),y_pred.detach().numpy(),color='orangered')
    plt.fill_between(x_test.detach().numpy().flatten(),y_pred.detach().numpy().flatten()-y_error.detach().numpy().flatten(),y_pred.detach().numpy().flatten()+y_error.detach().numpy().flatten(),color='orangered',alpha=0.5)
    plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.show()

