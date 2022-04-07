#==================================================================
# Copied from rnn_concat. Refer to pytorch/fnn.py to modify
#==================================================================


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

class fnn:
    def __init__(self,learning_rate=1.e-3,optimizer="rmsprop", n_restarts_optimizer=100,epsilon=1.e-5):
        self.learning_rate = learning_rate
        self.optimizer=optimizer
        self.n_restarts_optimizer=n_restarts_optimizer
        self.epsilon=epsilon
    
    def initialize(self,x_train,y_train,hidden_units=2,activation='tanh'):
        if len(x_train)>0:
            self.x_train = x_train.reshape(x_train.shape[0],-1)
        else:
            self.x_train = x_train
        self.y_train = y_train

        ######### Model parameters ###########
        self.x_size=self.x_train.shape[1]
        self.y_size=self.y_train.shape[1]
        self.batch_size = self.x_train.shape[0]
        self.h_size = hidden_units
        if(activation=='tanh'):
            self.activation = lambda x: np.tanh(x)
            self.dactivation = lambda y: (1.-y*y)
        elif(activation=='relu'):
            self.activation = lambda x: x*(x>0.)
            self.dactivation = lambda y: np.sign(y)
        else:
            print("Activation function not defined")
        
        #weights and biases
        self.parameters = OrderedDict()
        self.parameters['Wh'] = np.random.randn(self.h_size, self.x_size)*0.01 # h to x
        self.parameters['Wy'] = np.random.randn(self.y_size, self.h_size)*0.01 # h to y
        self.parameters['bh'] = np.zeros((self.h_size, 1)) # x bias
        self.parameters['by'] = np.zeros((self.y_size, 1)) # y bias
        self.total_parameters = 0
        for key in self.parameters.keys():
            self.total_parameters += self.parameters[key].size
        
    
    def print_dashed_line(self):
        print("--------------------------------------------")
    
    def get_parameters(self):
        self.print_dashed_line()
        print(" Parameters and dimensions ")
        self.print_dashed_line()
        for key in self.parameters.keys():
            print(str(key)+": (",self.parameters[key].shape[0],"x",self.parameters[key].shape[1],") - ",self.parameters[key].size," parameters")
        
        print("Total parameters = ",self.total_parameters)
        self.print_dashed_line()
        print(" Values ")
        self.print_dashed_line()
        for key in self.parameters.keys():
            print(str(key)+": ")
            self.print_dashed_line()
            print(self.parameters[key])
            self.print_dashed_line()
    
    def forward_prop(self,x,theta):
        parameters = OrderedDict(zip(self.parameters.keys(),theta))
        dparameters = OrderedDict()
        for key in parameters.keys():
            dparameters[key] = np.zeros_like(parameters[key])
        
        h = self.activation( parameters['Wh'] @ x.T + parameters['bh'])
        return (parameters['Wy'] @ h + parameters['by']).T,h

    def loss_function(self,x_train,y_train,theta):
        batch_size=x_train.shape[0]
        y_pred,h = self.forward_prop(x_train,theta)
        dy = 2*(y_train-y_pred).T
        loss = np.sum(np.power(dy/2.,2))/batch_size
        return loss,dy,h
    
    def backprop(self,x_train,y_train,theta):
        loss,dy,h = self.loss_function(x_train,y_train,theta)
        #unpack theta
        parameters = OrderedDict(zip(self.parameters.keys(),theta))
        dparameters = OrderedDict()
        for key in parameters.keys():
            dparameters[key] = np.zeros_like(parameters[key])
        
        batch_size=x_train.shape[0]
        
        x=x_train
        y=y_train
        
        #Calculate gradients dWxf, dbf, dWfx, dbx, dWgf, dWxg, dWgg, dbg
        dparameters['Wy'] += np.dot(dy, h.T)
        dparameters['by'] += np.dot(dy, np.ones((dy.shape[1],dparameters['by'].shape[1])))
        
        dh = np.dot(parameters['Wy'].T, dy) # backprop into h
        dz = self.dactivation(h) * dh # backprop through relu nonlinearity
        dparameters['Wh'] += np.dot(dz, x)
        dparameters['bh'] += np.dot(dz, np.ones((dz.shape[1],dparameters['bh'].shape[1])))
        return loss,np.array(list(dparameters.values()))/batch_size
        #===============================================================================================================
        # Modified till here
        #===============================================================================================================    
    
    def gradient_check(self,x_train,y_train,theta,epsilon):
        loss,grad = self.backprop(x_train,y_train,theta)
        numerical_grad=[]
        for i in range(grad.size):
            numerical_grad.append(np.copy(grad[i]))
        
        numerical_grad=np.array(numerical_grad)
        difference=0.
        for i in range(theta.size):
            for j in range(theta[i].shape[0]):
                for k in range(theta[i].shape[1]):
                    theta_plus, theta_minus = [], []
                    for l in range(theta.size):
                        theta_plus.append(np.copy(theta[l]))
                        theta_minus.append(np.copy(theta[l]))
                    
                    theta_plus=np.array(theta_plus)
                    theta_minus=np.array(theta_minus)
                    theta_plus[i][j,k] = theta[i][j,k]+epsilon
                    theta_minus[i][j,k] = theta[i][j,k]-epsilon
                    loss_plus,_,_ = self.loss_function(x_train,y_train,theta_plus)
                    loss_minus,_,_ = self.loss_function(x_train,y_train,theta_minus)
                    numerical_grad[i][j,k] = (loss_plus-loss_minus)/(2.*epsilon)
                    difference += np.abs(numerical_grad[i][j,k]-grad[i][j,k])

        return difference,numerical_grad,grad
    
    def train(self,x_train,y_train,initial_theta,train_validation_split=0.7,epochs=10000,restore_best_theta=True):
        split_point=int(np.round(train_validation_split*x_train.shape[0]))
        X = x_train[:split_point,...]
        X = X.reshape(X.shape[0],-1)
        print("Length of X:",X.shape[0])
        y = y_train[:split_point,:]
        if split_point<x_train.shape[0]:
            X_val = x_train[split_point:,...]
            X_val = X_val.reshape(X_val.shape[0],-1)
            print("Length of X_val:",X_val.shape[0])
            y_val = y_train[split_point:,:]
        else:
            X_val = np.zeros((0,x_train.shape[1]))
            y_val = np.zeros((0,y_train.shape[1]))
        if(X_val.shape[0]<10):
            restore_best_theta=False
            print("restore_best_theta has been set to false as validation set has less than ten data points")
        
        theta=np.array(initial_theta)
        best_loss_val=1.e100
        Eg2 = np.copy(theta*0)
        loss_vector=[]
        loss_val_vector=[]
        
        self.print_dashed_line()
        for i in range(epochs):
            
            (loss,grad) = self.backprop(X,y,theta)
            (loss_val,_,_) = self.loss_function(X_val,y_val,theta)
            
            loss_vector.append(loss)
            loss_val_vector.append(loss_val)
            if(restore_best_theta):
                if(loss_val<best_loss_val):
                    best_loss_val=loss_val
                    self.parameters = OrderedDict(zip(self.parameters.keys(),theta))
                
            else:
                self.parameters = OrderedDict(zip(self.parameters.keys(),theta)) 
            
            if i==0:
                Eg2 = np.power(grad,2)
            else:
                Eg2 = 0.9*Eg2 + 0.1*np.power(grad,2)
            
            delta_theta = self.learning_rate*grad/np.power(Eg2 + 1.e-8,0.5)
            theta = theta + delta_theta
            if(i%(epochs/10)==0):
                print("Iteration: ",i," Training loss: ",loss," Validation loss: ",loss_val)
                #print("Theta: ",np.hstack(theta[0]),np.hstack(theta[1]),np.hstack(theta[2]),np.hstack(theta[3]),np.hstack(theta[4]))
                #print("Grad_theta: ",np.hstack(grad[0]),np.hstack(grad[1]),np.hstack(grad[2]),np.hstack(grad[3]),np.hstack(theta[4]))
                #print("Delta_theta: ",np.hstack(delta_theta[0]),np.hstack(delta_theta[1]),np.hstack(delta_theta[2]),np.hstack(delta_theta[3]),np.hstack(delta_theta[4]))
                #self.print_dashed_line()
            if loss < 1.e-3:
                break
        
        return loss_vector, loss_val_vector
    
    def predict(self,x):
        x_reshaped = x.reshape(x.shape[0],-1)
        theta=np.asarray(list(self.parameters.values()))
        y_pred,_=self.forward_prop(x_reshaped,theta)
        return y_pred

