import torch
import torch.nn as nn
import torch.nn.functional as F
from core.fim_model import ModelFIM

class CNN(ModelFIM):
    def __init__(self, args):
        super(CNN, self).__init__(args)
        self.subspace_fraction = args.subspace_fraction
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout2d(0.25)
        #self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        super(CNN, self).common_init(args)

    def forward(self, x):
        self.a0 = x
        #print('a0 shape = {}'.format(self.a0.shape))
        self.s0 = self.conv1(self.a0)
        #print('s0 shape = {}'.format(self.s0.shape))
        self.a1 = F.relu(self.s0)
        #print('a1 shape = {}'.format(self.a1.shape))
        x = F.max_pool2d(self.a1, 2)
        self.s1 = self.conv2(x)
        #print('s1 shape = {}'.format(self.s1.shape))
        x = F.relu(self.s1)

        x = F.max_pool2d(x, 2)
        #print('max_pool2d output x shape = {}'.format(x.shape))
        #x = self.dropout1(x)
        self.a2 = torch.flatten(x, 1)
        #print('flatten output x shape = {}'.format(self.a2.shape))
        self.s2 = self.fc1(self.a2)
        #print('fc1 output x shape = {}'.format(self.s2.shape))
        self.a3 = F.relu(self.s2)
        #print('relu output x shape = {}'.format(self.a3.shape))
        #x = self.dropout2(self.a3)
        self.s3 = self.fc2(self.a3)
        #print('fc2 output x shape = {}'.format(self.s3.shape))
        output = F.log_softmax(self.s3, dim=1)
        #print('output x shape = {}'.format(output.shape))


        return output







