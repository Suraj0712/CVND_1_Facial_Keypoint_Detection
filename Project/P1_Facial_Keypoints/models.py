## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # nn.Conv2d(input_channels, output_channels, Kernal_size, padding)  
        # nn.MaxPool2d(Kernal_size, strides)
        # nn.BatchNorm2d(#of feature maps)
        # F.relu(pass the layer)
        # image\featuremap width\height = W, Kernal size = K, strides = S
        # Output size of layer after convoliution = ((W - K) / S)+1
        # x -> h1 -> h2 -> h3 -> h4 -> h5 -> fc1 -> fc2 -> O  
        # while going from I -> h1(conv1) till h5 [Convolution -> Relu Activation -> Batch norm -> Max pooling -> dropout]
        # While going from h5 -> fc1(fc1) till O -> [Fully connected -> relu]
        # Batch norm (bn1) -> normalised the values of featuremaps
 
        self.conv1 = nn.Conv2d(1, 16, 3)         #222*222*16
        self.pool1 = nn.MaxPool2d(2,2)           #111*111*16
        self.bn1 = nn.BatchNorm2d(16)            #111*111*16
        self.drop1 = nn.Dropout(p=0.2)           #111*111*16
        
        self.conv2 = nn.Conv2d(16, 32, 4)        #108*108*32
        self.pool2 = nn.MaxPool2d(2,2)           #54*54*32
        self.bn2 = nn.BatchNorm2d(32)            #54*54*32
        self.drop2 = nn.Dropout(p=0.2)           #54*54*32
 
        self.conv3 = nn.Conv2d(32, 64, 5)        #50*50*64
        self.pool3 = nn.MaxPool2d(2,2)           #25*25*64
        self.bn3 = nn.BatchNorm2d(64)            #25*25*64
        self.drop3 = nn.Dropout(p=0.2)           #25*25*32
        
        self.conv4 = nn.Conv2d(64, 128, 4)       #22*22*128
        self.pool4 = nn.MaxPool2d(2,2)           #11*11*128
        self.bn4 = nn.BatchNorm2d(128)           #11*11*128
        self.drop4 = nn.Dropout(p=0.2)           #11*11*32
        
        self.conv5 = nn.Conv2d(128, 256, 2)      #10*10*256
        self.pool5 = nn.MaxPool2d(2,2)           #5*5*256
        self.bn5 = nn.BatchNorm2d(256)           #5*5*256
        self.drop5 = nn.Dropout(p=0.2)           #5*5*256 = 6400
        
        self.fc1 = nn.Linear(5*5*256 , 2048)     #2048
        self.drop6 = nn.Dropout(p=0.2)           #2048
        
        self.fc2 = nn.Linear(2048 , 1024)        #1024
        self.drop7 = nn.Dropout(p=0.2)           #1024
        
        self.fc3 = nn.Linear(1024 , 136)         #136 = 2*68 key_points

        
    def forward(self, x):
        
        h1 = self.drop1(self.pool1(self.bn1(F.relu(self.conv1(x)))))
        h2 = self.drop2(self.pool2(self.bn2(F.relu(self.conv2(h1)))))
        h3 = self.drop3(self.pool3(self.bn3(F.relu(self.conv3(h2)))))
        h4 = self.drop4(self.pool4(self.bn4(F.relu(self.conv4(h3)))))
        h5 = self.drop5(self.pool5(self.bn5(F.relu(self.conv5(h4)))))
              
        h5 = h5.view(h5.size(0), -1)
               
        fc1 = self.drop6(F.relu(self.fc1(h5)))
        fc2 = self.drop7(F.relu(self.fc2(fc1)))
        output = (self.fc3(fc2))

        return output
