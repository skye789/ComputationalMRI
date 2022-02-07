#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST classification for Computational MRI 2021/2022

refer to https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
from __future__ import print_function
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import  optim
import os
import  numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.close("all")

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.FC_1hidden = torch.nn.Sequential(torch.nn.Linear(28 * 28, 30),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(30, 10),
                                             torch.nn.LogSoftmax(dim=1))

        self.FC_5hidden = torch.nn.Sequential(torch.nn.Linear(28 * 28 , 30),
                                              torch.nn.BatchNorm1d(30),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(30, 50),
                                              torch.nn.BatchNorm1d(50),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(50, 80),
                                              torch.nn.BatchNorm1d(80),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(80, 100),
                                              torch.nn.BatchNorm1d(100),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(100, 120),
                                              torch.nn.BatchNorm1d(120),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(120, 150),
                                              torch.nn.BatchNorm1d(150),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(150, 10),
                                               torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.FC_5hidden(x)
        return x


# %%Train network
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Epoch{}: \nTrain set : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch, sum_loss / len(train_loader.dataset),correct,
        len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return sum_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


# %% Calculate validation accuracy
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss / len(test_loader.dataset),correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


def visualize(args,y1,y2,type):
    '''''
    plot loss and accuracy along epoch
    '''
    x = np.arange(1, 1+args.epochs)
    y1 = y1.numpy()
    y2 = y2.numpy()
    plt.plot(x, y1, color='red', label='train')
    plt.plot(x, y2, color='blue', label='test')
    plt.title(type + ' (5 nidden_layer,' + ' ReLU)')
    plt.xlabel('Epoch')
    plt.ylabel(type)
    plt.legend([type+'--train', type+'--test'], loc='upper left')
    plt.show()


def main():
    # hyperparameter setting
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #load data
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #define model
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_loss = torch.zeros(args.epochs)
    train_acc = torch.zeros(args.epochs)
    test_loss = torch.zeros(args.epochs)
    test_acc = torch.zeros(args.epochs)

    # trainging and testing
    for epoch in range(1, args.epochs + 1):
        train_loss[epoch-1], train_acc[epoch-1] = train(args, model, device, train_loader, optimizer, epoch)
        test_loss[epoch-1], test_acc[epoch-1] = test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    # plot loss and accuracy
    visualize(args, train_loss, test_loss, type='Loss')
    visualize(args, train_acc, test_acc, type='Accuracy')



if __name__ == '__main__':
    main()