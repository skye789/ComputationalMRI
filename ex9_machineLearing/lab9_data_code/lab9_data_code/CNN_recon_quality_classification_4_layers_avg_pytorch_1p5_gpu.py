#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CNN for classification of compressed sensing 4 times accelerated vs fully sampled reference reconstructions
Didactic example to demonstrate model over- and underfitting for Computational MRI 2021/2022

Based on May 2018 for ISMRM educational "How to Jump-Start Your Deep Learning Research"
Educational course Deep Learning: Everything You Want to Know, Saturday, June 16th 2018
Joint Annual meeting of ISMRM and ESMRMB, Paris, France, June 16th to 21st

Created with PyTorch 1.5 and Python 3.7
Please see import section for module dependencies

florian.knoll@fau.de
"""
 
#%reset

#%% import packages
import numpy as np
np.random.seed(123)  # for reproducibility
import os

from sklearn.preprocessing import LabelEncoder
from imutils import paths
import cv2
from time import time
from matplotlib import pyplot as plt
import os,os.path
from rgb2gray import rgb2gray

import torch
import torch.nn as nn
import torch.utils.data as data_utils
torch.manual_seed(123)  # for reproducibility
 
plt.close("all")

#%% Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.get_device_name(device=None))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%% Paths and training parameters
path_base = "./data/recon_classification/";
# path_base = "/knolllab/knolllabspace/data/2017_07_27_pngs_Learning_Classification/";

recon1 = "ref"
# recon2 = "zf"
# recon2 = "cg"
recon2 = "tgv"
# recon2 = "learning"

print("paths for training and test image data")
path_base_training_recon1 = "{}{}{}".format(path_base,"train/",recon1,"/")
path_base_training_recon2 = "{}{}{}".format(path_base,"train/",recon2,"/")
path_base_testing_recon1 = "{}{}{}".format(path_base,"test/",recon1,"/")
path_base_testing_recon2 = "{}{}{}".format(path_base,"test/",recon2,"/")
# path_base_training_recon1 = "{}{}{}".format(path_base,"2017_MRM_paper_ref_ipat4_training_randomized_oneTenth/",recon1,"/")
# path_base_training_recon2 = "{}{}{}".format(path_base,"2017_MRM_paper_ref_ipat4_training_randomized_oneTenth/",recon2,"/")
# path_base_testing_recon1 = "{}{}{}".format(path_base,"2017_MRM_paper_ref_ipat4_testing_randomized_oneTenth/",recon1,"/")
# path_base_testing_recon2 = "{}{}{}".format(path_base,"2017_MRM_paper_ref_ipat4_testing_randomized_oneTenth/",recon2,"/")

trainingPaths = list(paths.list_images(path_base_training_recon1)) + list(paths.list_images(path_base_training_recon2))
testingPaths = list(paths.list_images(path_base_testing_recon1)) + list(paths.list_images(path_base_testing_recon2))

nTrainImages = len(trainingPaths)
nTestImages = len(testingPaths)

#%% Load data
# initialize data matrix and labels list
x_train  = []
y_train = []
x_test   = []
y_test  = []

# Training data
print("loading {} train images".format(nTrainImages))
for (i, imagePath) in enumerate(trainingPaths):
    image = cv2.imread(imagePath)
    image = rgb2gray(image)
    [nR,nC] = image.shape
    image = image.reshape(nR, nC, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    x_train.append(image)
    y_train.append(label)

    if i > 0 and i % 1000 == 0:
        print("train images processed {}/{}".format(i, nTrainImages))

# Random shuffling: Do this manually here
print("Shuffling training data")
idx_train_rand = np.random.permutation(nTrainImages)
x_train_rand   = []
y_train_rand = []
for i in range(0, nTrainImages):
    # print("i: {}".format(i))
    x_train_rand.append(x_train[idx_train_rand[i]])
    y_train_rand.append(y_train[idx_train_rand[i]])

x_train= x_train_rand
del x_train_rand
y_train = y_train_rand
del y_train_rand

# In case we want to plot one of our images
# plt.figure(1)
# plt.imshow(x_train[1][0,:,:],cmap="gray")
# plt.axis('off')
# plt.title(y_train[1])

# Test data
print("loading {} test images".format(nTestImages))
for (i, imagePath) in enumerate(testingPaths):
    image = cv2.imread(imagePath)
    image = rgb2gray(image)
    [nR,nC] = image.shape
    image = image.reshape(nR, nC, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    x_test.append(image)
    y_test.append(label)

    if i > 0 and i % 1000 == 0:
        print("test images processed {}/{}".format(i, nTestImages))
        
# In case we want to plot one of our test images
# plt.figure(2)
# plt.imshow(x_test[100][:,:,0],cmap="gray")
# plt.axis('off')
# plt.title(y_test[100])

#%% encode the labels, converting them from strings to integers
le = LabelEncoder()
y_train= le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# scale the input image pixels to the range [0,1]
x_train= np.array(x_train)
x_test= np.array(x_test)

print("Normalizing images to mean 0 and std 1")
for i in range(0, nTrainImages):
    # x_train[i,:,:,:] = x_train[i,:,:,:] / 255
    # print("Train image {} min {} max {}".format(i, np.min(x_train[i,:,:,:]), np.max(x_train[i,:,:,:])))
    x_train[i,:,:,:] = x_train[i,:,:,:] - np.mean(x_train[i,:,:,:])
    x_train[i,:,:,:] = x_train[i,:,:,:] / np.std(x_train[i,:,:,:])
    # print("Train image {} mean {} std {}".format(i, np.mean(x_train[i,:,:,:]), np.std(x_train[i,:,:,:])))
    
for i in range(0, nTestImages):
    # x_test[i,:,:,:] = x_test[i,:,:,:] / 255
    # print("Test image {} min {} max {}".format(i, np.min(x_test[i,:,:,:]), np.max(x_test[i,:,:,:])))
    x_test[i,:,:,:] = x_test[i,:,:,:] - np.mean(x_test[i,:,:,:])
    x_test[i,:,:,:] = x_test[i,:,:,:] / np.std(x_test[i,:,:,:])
    # print("Test image {} mean {} std {}".format(i, np.mean(x_test[i,:,:,:]), np.std(x_test[i,:,:,:])))
    
# image = image.reshape(nR, nC, 1)

#%% Reshape the 3d array of images
x_train = x_train.reshape(nTrainImages,1,nR,nC)
x_test = x_test.reshape(nTestImages,1,nR,nC)

#%%Generate torch variables 
x_train = torch.Tensor(x_train).float()
y_train = torch.Tensor(y_train).long()

x_test = torch.Tensor(x_test).float()
y_test = torch.Tensor(y_test).long()

#%% Define model
model_name = "CNN4layers_global_avg"
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,4,(3,3),padding=(1,1))
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4,4,(3,3),padding=(1,1))
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(4,4,(3,3),padding=(1,1))
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(4,4,(3,3),padding=(1,1))
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.out = nn.Linear(4,2)
        # self.Sigmoid = nn.Sigmoid()
        # self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        # print("Size after first conv: {}".format(x.shape))
        x = self.pool(x)
        # print("Size after first pool: {}".format(x.shape))
        
        x = self.ReLU(self.conv2(x))
        x = self.pool(x)
        # print("Size after second pool: {}".format(x.shape))
         
        x = self.ReLU(self.conv2(x))
        x = self.pool(x)
        # print("Size after third pool: {}".format(x.shape))
         
        x = self.ReLU(self.conv2(x))
        x = self.pool(x)
        # print("Size after fourth pool: {}".format(x.shape))
        
        # global average pooling 2d
        x = x.mean([2, 3])
        # print("Size after global average pool: {}".format(x.shape))
        
        # x = self.Sigmoid(self.out(x))
        # x = self.Softmax(self.out(x))
        x = self.LogSoftmax(self.out(x))
        # print("Size after output layer: {}".format(x.shape))
        
        return x

#%% Generate instance of model
model = Model()
print(model)
model.to(device)
 
#%%choose optimizer and loss function
training_epochs = 1000 # 1000
lr = 0.001
batch_size = 10 # 390
# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#%%Create minibatch data loading for training and validation
dataloader_train = data_utils.TensorDataset(x_train, y_train)
dataloader_train = data_utils.DataLoader(dataloader_train, batch_size=batch_size, shuffle=False,num_workers=4)

#%% Train model
loss_train = np.zeros(training_epochs)
acc_train = np.zeros(training_epochs)
loss_test = np.zeros(training_epochs)
acc_test = np.zeros(training_epochs)

time0 = time()
for epoch in range(training_epochs):
    for local_batch, local_labels in dataloader_train:
        # feedforward - backpropagation
        optimizer.zero_grad()
        out = model(local_batch.cuda())
        loss = criterion(out, local_labels.cuda())
        loss.backward()
        optimizer.step()
        loss_train[epoch] = loss.cpu().item()
        # Training data accuracy
        # [dummy, predicted] = torch.max(out.data, 1)
        # acc_train[epoch] = (torch.sum(local_labels==predicted.cpu()).numpy() / np.size(local_labels.numpy(),0))
 
    # Full training set accuracy
    out_train = model(x_train.cuda())
    loss = criterion(out_train, y_train.cuda())
    loss_train[epoch] = loss.cpu().item()
    [dummy, predicted_train] = torch.max(out_train.data, 1)
    acc_train[epoch] = (torch.sum(y_train==predicted_train.cpu()).numpy() /nTrainImages)
    
    # Full validation set accuracy
    out_test = model(x_test.cuda())
    loss = criterion(out_test, y_test.cuda())
    loss_test[epoch] = loss.cpu().item()
    [dummy, predicted_test] = torch.max(out_test.data, 1)
    acc_test[epoch] = ( torch.sum(y_test==predicted_test.cpu()).numpy() /nTestImages)
    
    print ('Epoch {}/{} train loss: {:.3}, train acc: {:.3}, val loss: {:.3}, val acc: {:.3}'.format(epoch+1, training_epochs, loss_train[epoch], acc_train[epoch], loss_test[epoch], acc_test[epoch]))

print("\nTraining Time (in minutes) =",(time()-time0)/60,"\n")

#%% Evaluate trained model
# Double check model on train data
out = model(x_train.cuda())
[dummy, predicted_train] = torch.max(out.data, 1)
acc_train_final = (torch.sum(y_train==predicted_train.cpu()).numpy() / nTrainImages)
print('Evaluation results train data: loss {:.2} acc {:.2}'.format(loss_train[training_epochs-1],acc_train_final))

# Evaluate model on test data
out_test = model(x_test.cuda())
[dummy, predicted_test] = torch.max(out_test.data, 1)
acc_test_final = (torch.sum(y_test==predicted_test.cpu()).numpy() / nTestImages)
print('Evaluation results test data: loss {:.2} acc {:.2}'.format(loss_test[training_epochs-1],acc_test_final))
 
#%% Evaluate predicted labels
# predictedLabels_round = np.round(predicted_test)
# predictedLabels_round = np.squeeze(np.transpose(predictedLabels_round.astype(int)))
# predictedLabels_name = np.round(predicted_test)
# predictedLabels_name = predictedLabels_name.astype(int)
# predictedLabels_name = le.inverse_transform(predictedLabels_name)  
# predictedLabels_name = np.squeeze(predictedLabels_name)
# testLabels_name = le.inverse_transform(y_test)  
# correctClassifications = np.transpose(y_test==predictedLabels_round)

#%% Print model's state_dict
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
#print("Optimizer's state_dict:")
#for var_name in optimizer.state_dict():
#    print(var_name, "\t", optimizer.state_dict()[var_name])
    
#%% Save trained model state_dic
#torch.save(model.state_dict(), "./trained_models/" + model_name + "_epochs10.pt")
#torch.save(optimizer.state_dict(), "./trained_models/" + model_name + "_optimizer_epochs10.pt")

#%% Plot convergence
if model_name == "CNN1layers_global_avg":
    # plot_label = "1 conv layer global average"
    plot_label_train = '1 conv layer global average: train={:.2}'.format(acc_train_final)
    plot_label_train_val = '1 conv layer global average: train/val={:.2}/{:.2}'.format(acc_train_final,acc_test_final)
elif model_name == "CNN4layers_global_avg":
    # plot_label = "4 conv layers global average"
    plot_label_train = '4 conv layers global average: train={:.2}'.format(acc_train_final)
    plot_label_train_val = '4 conv layers global average: train/val={:.2}/{:.2}'.format(acc_train_final,acc_test_final)
elif model_name == "CNN4layers_FC":
     # plot_label = "4 conv layers fully connected"
     plot_label_train = '4 conv layers fully connected: train={:.2}'.format(acc_train_final)
     plot_label_train_val = '4 conv layers fully connected: train/val={:.2}/{:.2}'.format(acc_train_final,acc_test_final)
else:
     print('This model does not exist')

N=1
plt.figure(1)
plt.plot(np.convolve(acc_train, np.ones((N,))/N, mode='valid'))
plt.title(plot_label_train)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training'], loc='lower right')
plt.ylim(0.4,1.1)
#plt.show()
plt.savefig('./training_plots_pytorch/{}_{}_{}_{}_train.png'.format(model_name,training_epochs,recon1,recon2))

plt.figure(2)
plt.plot(np.convolve(acc_train, np.ones((N,))/N, mode='valid'))
plt.plot(np.convolve(acc_test , np.ones((N,))/N, mode='valid'))
plt.title(plot_label_train_val)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='lower right')
plt.ylim(0.4,1.1)
#plt.show()
plt.savefig('./training_plots_pytorch/{}_{}_{}_{}_train_val.png'.format(model_name,training_epochs,recon1,recon2))

N=10
plt.figure(3)
plt.plot(np.convolve(acc_train, np.ones((N,))/N, mode='valid'))
plt.title(plot_label_train)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training'], loc='lower right')
plt.ylim(0.4,1.1)
#plt.show()
plt.savefig('./training_plots_pytorch/{}_{}_{}_{}_train_maconv.png'.format(model_name,training_epochs,recon1,recon2))

plt.figure(4)
plt.plot(np.convolve(acc_train, np.ones((N,))/N, mode='valid'))
plt.plot(np.convolve(acc_test , np.ones((N,))/N, mode='valid'))
plt.title(plot_label_train_val)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='lower right')
plt.ylim(0.4,1.1)
#plt.show()
plt.savefig('./training_plots_pytorch/{}_{}_{}_{}_train_val_maconv.png'.format(model_name,training_epochs,recon1,recon2))

