#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch linear regression demo

Based on May 2018 ISMRM educational "How to Jump-Start Your Deep Learning Research"
Educational course Deep Learning: Everything You Want to Know, Saturday, June 16th 2018
Joint Annual meeting of ISMRM and ESMRMB, Paris, France, June 16th to 21st

Created with PyTorch 1.5 and Python 3.7
Please see import section for module dependencies

Due to the small data and to make it available for systems without a GPU, this example was created to run on the CPU
If you want to run it on the GPU, simply create tensors on the GPU, e.g.:
x_train = torch.Tensor([1,2,3,4,5]).float().to(torch.device("cuda"))
...
However, for this oversimplified example, the GPUs are not utilized particularly well,
and there is a significant overhead just due to the display of the parameters and the loss over the epochs.

florian.knoll@fau.de

"""
#%reset

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

#%% training data
x_train = torch.Tensor([1,2,3,4,5]).float()
y_train = torch.Tensor([3,5,7,9,11]).float()

#%% model parameters
k = torch.Tensor([0.1]).float()
k.requires_grad=True

d = torch.Tensor([-0.1]).float()
d.requires_grad=True

#%% model and optimizer
def forward(x):
    return x * k + d
 
def loss_fcn(x,y):
    y_pred = forward(x)
    return (y_pred-y) * (y_pred-y)

#%% train
training_epochs = 1000
lr = 0.005
loss_ii = np.zeros(training_epochs)
k_ii = np.zeros(training_epochs)
d_ii = np.zeros(training_epochs)

t = time.time()
for ii in range(training_epochs):
    for x_train_batch, y_train_batch in zip(x_train,y_train):
        loss = loss_fcn(x_train_batch, y_train_batch)
        loss_ii[ii] = loss.item()
        k_ii[ii] = k.data
        d_ii[ii] = d.data
        loss.backward()
        k.data = k.data - lr * k.grad.data
        d.data = d.data - lr * d.grad.data
        
        # Manually set gradient to zero after update step to prevent gradient accumulation
        k.grad.data.zero_()
        d.grad.data.zero_()
    print('epoch: {}, k={:.3}, d={:.3}, loss={:.3}'.format(ii+1,k_ii[ii], d_ii[ii], loss_ii[ii]))
    
elapsed = time.time() - t
print('Training time: {:.2} s'.format(elapsed))

#%% Plot training overview
plt.figure(1)
plt.plot(loss_ii)
plt.title('Linear regression loss')
plt.ylabel('SOS error (a.u.)')
plt.xlabel('epoch')
plt.show()
plt.savefig('./training_plots_pytorch/linear_regression_epochs{}.png'.format(training_epochs))

#%% Plot training overview
plt.figure(2)
plt.plot(k_ii)
plt.plot(d_ii)
plt.title('Trained model parameters')
plt.ylabel('parameter values')
plt.xlabel('epoch')
plt.legend(['k', 'd'], loc='lower right')
plt.show()
plt.savefig('./training_plots_pytorch/linear_regression_model_parameters_epochs{}.png'.format(training_epochs))