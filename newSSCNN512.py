import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms

import time
import torch.optim as optim

EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.003

#TRAIN_DATA_PATH = "/content/drive/My Drive/school/thesis/SecondaryStructureClassificationPytorch/proteinsTrain275"
#TEST_DATA_PATH = "/content/drive/My Drive/school/thesis/SecondaryStructureClassificationPytorch/proteinsTest40"
TRAIN_DATA_PATH = "./proteinsTrain275"
TEST_DATA_PATH = "./proteinsTest40"

CLASS_A = "allalpha"
CLASS_B = "allbeta"
CLASS_C = "mixedalphabeta"
CLASS_D = "segregatedalphabeta"


#TRAIN_DATA_PATH = "/content/drive/My Drive/SecondaryStructureClassificationPytorch/proteinsTrain275"
#TEST_DATA_PATH = "/content/drive/My Drive/SecondaryStructureClassificationPytorch/proteinsTest40"

TRANSFORM_IMG = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.Resize((512, 512)),
        #transforms.CenterCrop(256),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] )
])

#training data
train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

#validation data
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)
  
def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    print("Number of train samples: ", len(train_data))
    print("Number of test samples: ", len(test_data))
    print("Detected Classes are: ", train_data.class_to_idx)
    
    #Get training data
    train_loader = train_data_loader #get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            #running_loss += loss_size.data[0]
            #total_train_loss += loss_size.data[0]
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                print("this is the {} th running".format(i))
                running_loss = 0.0
                start_time = time.time()
            
        print("epoch finished, took {:.2f}s".format(time.time() - training_start_time))
        #At the end of the epoch, do a pass on the validation set
        computeAccuracy(net, loss, test_data_loader, 'validate epoch')
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    computeAccuracy(net, loss, train_data_loader, 'training data')
  
    computeAccuracy(net, loss, test_data_loader, 'test epoch')
    

def computeAccuracy(net, loss, accuracy_data_loader, title):
  total_val_loss = 0
  total = 0
  correct = 0
  
  total_a = 0
  correct_a = 0
  total_b = 0
  correct_b = 0
  total_c = 0
  correct_c = 0
  total_d = 0
  correct_d = 0
  
  for inputs, labels in accuracy_data_loader:

      #Wrap tensors in Variables
      inputs, labels = Variable(inputs), Variable(labels)

      #Forward pass
      val_outputs = net(inputs)
      val_loss_size = loss(val_outputs, labels)

      _, predicted = torch.max(val_outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      
      for l, p in zip(labels, predicted):
          if l.item() == 0:
            total_a +=1
            if l == p:
              correct_a +=1
          elif l.item() == 1:
            total_b +=1
            if l == p:
              correct_b +=1
          elif l.item() == 2:
            total_c +=1
            if l == p:
              correct_c +=1
          elif l.item() == 3:
            total_d +=1
            if l == p:
              correct_d +=1

      #total_val_loss += val_loss_size.data[0]
      total_val_loss += val_loss_size.item()

  print("Validation loss = {:.2f}".format(total_val_loss / len(accuracy_data_loader)))

  print("{} total images {}".format(title, total))
  print("{} correct images {}".format(title, correct))
  
  print('Accuracy of the network on the {} images: {} %%'.format(title, 100 * correct / total))
  
  print("total a images {}".format(total_a))
  print("correct a images {}".format(correct_a))
  print('Accuracy of the a images: {} %%'.format(100 * correct_a / total_a))
  
  print("total b images {}".format(total_b))
  print("correct b images {}".format(correct_b))
  print('Accuracy of the b images: {} %%'.format(100 * correct_b / total_b))
  
  print("total c images {}".format(total_c))
  print("correct c images {}".format(correct_c))
  print('Accuracy of the c images: {} %%'.format(100 * correct_c / total_c))
  
  print("total d images {}".format(total_d))
  print("correct d images {}".format(correct_d))
  print('Accuracy of the d images: {} %%'.format(100 * correct_d / total_d))
  
class CNN(nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        #Input channels = 18, output channels = 64
        self.conv2 = torch.nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        
        
        #4608 input features, 64 output features (see sizing flow below)
        #self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc1 = torch.nn.Linear(512*16*16, 64)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 4)
        
    def forward(self, x):
              
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv2(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        #x = x.view(-1, 18 * 16 *16)
        x = x.view(-1, 512*16*16)
          
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)

if __name__ == '__main__':
  print('starting main')
  CNN = CNN()
  trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)

