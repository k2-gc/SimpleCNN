import os
import torch
import torch.nn as nn
import torchvision  
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import model

FILE_PATH = "PATH TO FILE"
MODEL_PATH = "PATH TO MODEL"
image_size = 32
batch_size = 1
num_classes = 10

transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(), 
                                    ])
#torchvision.datasets.MNIST(os.path.join(FILE_PATH, 'MNIST'),
#                           train=True,
#                           transform=transform, 
#                           target_transform=transform, 
#                           download=True)
 
traindataset = torchvision.datasets.CIFAR10(os.path.join(FILE_PATH, 'CIFAR10'),
                                            train=False,
                                            transform=transform,
                                            target_transform=None,
                                            download=True)

shuffle = False
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=shuffle)

num_datasets = len(trainloader)
print("num of dataset:" + str(num_datasets))

net = model.SimpleCNN(3, num_classes)
net.load_state_dict(torch.load(MODEL_PATH))
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

max = 0
softmax = nn.Softmax()
for image_num, (images, labels) in enumerate(trainloader):
	pred = net(images)
	pred = softmax(pred)
	print(pred)
	print(torch.argmax(pred))
	pred = torch.argmax(pred, dim=1)
	pred = pred.unsqueeze(0)
	print(pred.shape)
	text_range = np.ones([3, images.shape[3], images.shape[1]]) * 255
	text_range = np.asarray(text_range, np.uint8)
	
	num = zip(pred, labels)	
	idx = 0
	for n, image in zip(num, images):	
		image = image.numpy() * 255
		image = np.asarray(image, np.uint8)
		image = image.transpose(1, 2, 0)
		image = np.concatenate([text_range, image], axis=0)
		plt.imshow(image)
		plt.text(0, 2, 'true: ' + str(n[0].item()) + ' predict:' + str(n[1].item()))
		#plt.show()
		plt.savefig('./images/pred_{}_{}.png'.format(image_num, idx))
        
    
