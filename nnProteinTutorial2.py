import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms

EPOCHS = 2
BATCH_SIZE = 10
LEARNING_RATE = 0.003
TRAIN_DATA_PATH = "./proteinsTrain275"
TEST_DATA_PATH = "./proteinsTest40"
TRANSFORM_IMG = transforms.Compose([
        #transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] )
])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':

   print("Number of train samples: ", len(train_data))
   print("Number of test samples: ", len(test_data))
   print("Detected Classes are: ", train_data.class_to_idx)
   # classes are detected by folder structure
   model = CNN()
   optimizer = torch.optim.Adam(model.parameters(),
   lr=LEARNING_RATE)
   loss_func = nn.CrossEntropyLoss()

   # Training and Testing
   for epoch in range(EPOCHS):
       for step, (x, y) in enumerate(train_data_loader):
           b_x = Variable(x)
           # batch
           # x
           # (image)
           b_y = Variable(y)
           # batch
           # y
           # (target)
           output = model(b_x)[0]
           loss = loss_func(output, b_y)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           if step % 50 == 0:
              test_x = Variable(test_data_loader)
              test_output, last_layer = model(test_x)
              pred_y = torch.max(test_output, 1)[1].data.squeeze()
              accuracy = sum(pred_y == test_y) / float(test_y.size(0))
              print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
