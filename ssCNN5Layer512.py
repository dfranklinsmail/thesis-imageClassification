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
TRAIN_DATA_PATH = "./proteins6NewSorted25PDB/training"
VALIDATION_DATA_PATH = "./proteins6NewSorted25PDB/validating"
TEST_DATA_PATH = "./proteins6NewSorted25PDB/testing"

D8244_DATA_PATH = "./proteinsD8244"
FC699_DATA_PATH = "./proteinsFC699"
D1185_DATA_PATH = "./proteinsD1185"

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
validation_data = torchvision.datasets.ImageFolder(root=VALIDATION_DATA_PATH, transform=TRANSFORM_IMG)
validation_data_loader  = data.DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#test data
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#D8244 data
d8244_data = torchvision.datasets.ImageFolder(root=D8244_DATA_PATH, transform=TRANSFORM_IMG)
d8244_data_loader  = data.DataLoader(d8244_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#D1185 data
d1185_data = torchvision.datasets.ImageFolder(root=D1185_DATA_PATH, transform=TRANSFORM_IMG)
d1185_data_loader  = data.DataLoader(d1185_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#FC699 data
fc699_data = torchvision.datasets.ImageFolder(root=FC699_DATA_PATH, transform=TRANSFORM_IMG)
fc699_data_loader  = data.DataLoader(fc699_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

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
    print("Number of test samples: ", len(validation_data))
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
        computeAccuracy(net, loss, validation_data_loader, 'validate epoch')
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    computeAccuracy(net, loss, train_data_loader, 'training data')
    computeAccuracy(net, loss, validation_data_loader, 'validate data')
    computeAccuracy(net, loss, test_data_loader, 'test epoch')
   
    computeAccuracy(net, loss, d8244_data_loader, 'D8244 dataset')
    #computeAccuracy(net, loss, d1185_data_loader, 'D1185 dataset')
    #computeAccuracy(net, loss, fc699_data_loader, 'FC699 dataset')

# def computeAccuracy(net, loss, accuracy_data_loader, title):
#   total_val_loss = 0
#   total = 0
#   correct = 0
#   for inputs, labels in accuracy_data_loader:

#       #Wrap tensors in Variables
#       inputs, labels = Variable(inputs), Variable(labels)

#       #Forward pass
#       val_outputs = net(inputs)
#       val_loss_size = loss(val_outputs, labels)

#       _, predicted = torch.max(val_outputs.data, 1)
#       total += labels.size(0)
#       correct += (predicted == labels).sum().item()

#       #total_val_loss += val_loss_size.data[0]
#       total_val_loss += val_loss_size.item()

#   print("Validation loss = {:.2f}".format(total_val_loss / len(accuracy_data_loader)))

#   print("{} total images {}".format(title, total))
#   print("{} correct images {}".format(title, correct))
  
#   print('Accuracy of the network on the {} images: {} %%'.format(title, 100 * correct / total))

def computeAccuracy(net, loss, accuracy_data_loader, title):
  total_val_loss = 0
  total = 0
  correct = 0
  
  total_a = 0
  correct_a = 0
  total_a_b = 0
  total_a_c = 0
  total_a_d = 0
  total_b = 0
  correct_b = 0
  total_b_a = 0
  total_b_c = 0
  total_b_d = 0
  total_c = 0
  correct_c = 0
  total_c_a = 0
  total_c_b = 0
  total_c_d = 0
  total_d = 0
  correct_d = 0
  total_d_a = 0
  total_d_b = 0
  total_d_c = 0
  
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
             elif p == 1:
                total_a_b +=1
             elif p == 2:
                total_a_c +=1
             elif p == 3:
                total_a_d +=1
          elif l.item() == 1:
             total_b +=1
             if l == p:
                correct_b +=1
             elif p == 0:
                total_b_a +=1
             elif p == 2:
                total_b_c +=1
             elif p == 3:
                total_b_d +=1
          elif l.item() == 2:
             total_c +=1
             if l == p:
                correct_c +=1
             elif p == 0:
                total_c_a +=1
             elif p == 1:
                total_c_b +=1
             elif p == 3:
                total_c_d +=1
          elif l.item() == 3:
             total_d +=1
             if l == p:
                correct_d +=1
             elif p == 0:
                total_d_a +=1
             elif p == 1:
                total_d_b +=1
             elif p ==2:
                total_d_c +=1
      #total_val_loss += val_loss_size.data[0]
      total_val_loss += val_loss_size.item()

  print("Validation loss = {:.2f}".format(total_val_loss / len(accuracy_data_loader)))

  print("{} total images {}".format(title, total))
  print("{} correct images {}".format(title, correct))
  
  print('Accuracy of the network on the {} images: {} %%'.format(title, 100 * correct / total))
  
  print("total a images {}".format(total_a))
  print("correct a images {}".format(correct_a))
  print('Accuracy of the a images: {} %%'.format(100 * correct_a / total_a))
  print("incorrect a images predicted b {}".format(total_a_b))
  print("incorrect a images predicted c {}".format(total_a_c))
  print("incorrect a images predicted d {}".format(total_a_d))
  
  print("total b images {}".format(total_b))
  print("correct b images {}".format(correct_b))
  print('Accuracy of the b images: {} %%'.format(100 * correct_b / total_b))
  print("incorrect b images predicted a {}".format(total_b_a))
  print("incorrect b images predicted c {}".format(total_b_c))
  print("incorrect b images predicted d {}".format(total_b_d))
  
  print("total c images {}".format(total_c))
  print("correct c images {}".format(correct_c))
  print('Accuracy of the c images: {} %%'.format(100 * correct_c / total_c))
  print("incorrect c images predicted a {}".format(total_c_a))
  print("incorrect c images predicted b {}".format(total_c_b))
  print("incorrect c images predicted d {}".format(total_c_d))
  
  print("total d images {}".format(total_d))
  print("correct d images {}".format(correct_d))
  print('Accuracy of the d images: {} %%'.format(100 * correct_d / total_d))
  print("incorrect d images predicted a {}".format(total_d_a))
  print("incorrect d images predicted b {}".format(total_d_b))
  print("incorrect d images predicted c {}".format(total_d_c))
  
class CNN5Layer(nn.Module):
    
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

class CNN(nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(CNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
         #Input channels = 18, output channels = 64
        self.conv2 = torch.nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #1048576 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(64 * 128 * 128, 64)
        
        #64 input features, 4 output features for our 4 defined classes
        self.fc2 = torch.nn.Linear(64, 4)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 512, 512) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool1(x)
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv2(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool2(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        #x = x.view(-1, 18 * 16 *16)
        x = x.view(-1, 64 * 128 * 128)
          
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)

if __name__ == '__main__':
  CNN = CNN()
  print(torch.cuda.is_available())
  trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)