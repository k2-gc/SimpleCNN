import os
import torch
import torch.nn as nn
import torchvision  
from torchvision.transforms import Compose
import numpy as np
from PIL import Image

import model

FILE_PATH = "PATH to FILE"
image_size = 32
batch_size = 1
num_classes = 10

transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(), 
                                    ])
torchvision.datasets.MNIST(os.path.join(FILE_PATH, 'MNIST'),
                           train=True,
                           transform=transform, 
                           target_transform=transform, 
                           download=True)
 
#traindataset = torchvision.datasets.CIFAR10(os.path.join(FILE_PATH, './CIFAR10/'),
#                                            train=True,
#                                            transform=transform,
#                                            target_transform=None,
#                                            download=True)

shuffle = False
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=shuffle)

num_datasets = len(trainloader)
print("num of dataset:" + str(num_datasets))

net = model.SimpleCNN(3, num_classes)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

loss_func = nn.CrossEntropyLoss()

max = 0
epochs = 100
loss_list = []
for epoch in range(epochs):
    for images, labels in trainloader:
        pred = net(images)
        loss = loss_func(pred, labels)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    print("epoch:{}, loss:{}".format((epoch+1), sum(loss_list)/num_datasets))
    loss_list = []
        
    
