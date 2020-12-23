# Sohan Gadiraju
# Programming Assignment 3
# Part 1a

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms 
import time
import os
import copy

num_epochs = 10
batch_size = 16
learning_rate = 0.001

data_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

trainset = torchvision.datasets.CIFAR100("data", train = True, transform = data_transforms, download=True)
testset = torchvision.datasets.CIFAR100("data", train=False, transform = data_transforms, download=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)

model = models.vgg16(pretrained = True)

num_ftrs = model.classifier[6].in_features
num_cls = 100;
model.classifier[6] = nn.Linear(num_ftrs, num_cls)

for param in model.parameters(): # freeze 
    param.requires_grad = False
for param in model.classifier[6].parameters(): # train the last linear layer. param.requires_grad = True
    param.requires_grad = True

num_epochs = 10
best_acc = 0.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
total_step = len(train_loader)
fail_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        fail_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

# Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Overall Test Accuracy: {} %'.format((correct / total) * 100))

# Saving
best_model_wts = copy.deepcopy(vgg16.state_dict())
torch.save(best_model_wts)


