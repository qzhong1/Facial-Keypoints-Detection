## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # first conv layer :input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input size 224
        # output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        self.conv1 = nn.Conv2d(1, 32, 3)
        
        # Maxpooling 5x5, stride = 5
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        # input size 220
        # output size = (W-F)/S +1 = (220-5)/1 +1 = 116
        # output after maxpooling size = 116/5 = 23 
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # third conv layer: 64 inputs, 64 outputs, 5x5 conv
        # input size 23
        # output size = (W-F)/S +1 = (23-5)/1 +1 = 19
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # fourth conv layer: 64 inputs, 64 outputs, 5x5 conv
        # input size 19
        # output size = (W-F)/S +1 = (220-5)/1 +1 = 15
        # output after maxpooling size = 15/5 = 3 
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # batchnormalisation
        self.batch1 = nn.BatchNorm2d(32)

        self.batch2 = nn.BatchNorm2d(64)

        self.batch3 = nn.BatchNorm2d(128)

        self.batch4 = nn.BatchNorm2d(256)
        
        # 500 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(256*12*12, 5000)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.3)
        
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(5000, 500)
        
        self.fc3 = nn.Linear(500, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batch3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.batch4(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc1_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
