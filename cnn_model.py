from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from fim_model import ModelFIM
import numpy as np


class CNN(ModelFIM):
    def __init__(self, subspace_fraction=0.1):
        super(CNN, self).__init__()
        self.subspace_fraction = subspace_fraction
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.conv_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.GS = OrderedDict()
        self.GS['PSI0_AVG'] = np.eye((1*3*3))
        self.GS['GAM0_AVG'] = np.eye((32))
        self.GS['PSI1_AVG'] = np.eye((32*3*3))
        self.GS['GAM1_AVG'] = np.eye((64))
        self.GS['PSI2_AVG'] = np.eye((1600))
        self.GS['GAM2_AVG'] = np.eye((128))
        self.GS['PSI3_AVG'] = np.eye((128))
        self.GS['GAM3_AVG'] = np.eye((10))
        #self.GS['PSI4_AVG'] = np.eye((128))
        #self.GS['GAM4_AVG'] = np.eye((10))

        self.GSLOWER = {}
        self.GSLOWERINV = {}
        for key, val in self.GS.items():
            self.GSLOWER[key] = np.eye(self.get_subspace_size(self.GS[key].shape[0]))
            self.GSLOWERINV[key] = np.eye(self.get_subspace_size(self.GS[key].shape[0]))
        self.GSINV = {}
        self.P = {}
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.GSINV[key] = self.GS[key]
            subspace_size = self.get_subspace_size(self.GSINV[key].shape[0])
            eigvec_subspace = self.GS[key][:, -subspace_size:]
            self.P[key] = eigvec_subspace

        self.corr_curr = [None]*len(self.GS)
        self.corr_curr_lower_proj = [None] * len(self.GS)
        self.corr_curr_lower = [None] * len(self.GS)
        self.tick = 0
        self.a0 = None
        self.s0 = None
        self.a1 = None
        self.s1 = None
        self.s2 = None
        self.a2 = None
        self.s3 = None
        self.a3 = None
        self.s4 = None

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
        x = self.dropout1(x)
        self.a2 = torch.flatten(x, 1)
        #print('flatten output x shape = {}'.format(self.a2.shape))
        self.s2 = self.fc1(self.a2)
        #print('fc1 output x shape = {}'.format(self.s2.shape))
        self.a3 = F.relu(self.s2)
        #print('relu output x shape = {}'.format(self.a3.shape))
        x = self.dropout2(self.a3)
        self.s3 = self.fc2(x)
        #print('fc2 output x shape = {}'.format(self.s3.shape))
        output = F.log_softmax(self.s3, dim=1)
        #print('output x shape = {}'.format(output.shape))
        self.s0.retain_grad()
        self.s1.retain_grad()
        self.s2.retain_grad()
        self.s3.retain_grad()

        return output


    def get_grads(self):
        a0 = self.a0.detach().numpy()
        s0_grad = self.s0.grad.detach().numpy()
        a1 = self.a1.detach().numpy()
        s1_grad = self.s1.grad.detach().numpy()
        a2 = self.a2.detach().numpy()
        s2_grad = self.s2.grad.detach().numpy()
        a3 = self.a3.detach().numpy()
        s3_grad = self.s3.grad.detach().numpy()
        # print('a0.shape = {}, so_grad.shape = {}, a1.shape = {}, s1_grad.shape = {}, a2.shape = {}, s2_grad.shape = {}'.format(a0.shape, s0_grad.shape, a1.shape, s1_grad.shape, a2.shape, s2_grad.shape))
        params = [a0, s0_grad, a1, s1_grad, a2, s2_grad, a3, s3_grad]
        param_out = []
        for itr, param in enumerate(params):
            if len(param.shape) == 4 :
                #print('itr={}: in param_shape = {}'.format(itr, param.shape))
                kernel_size = self.conv_layers[itr//2].kernel_size
                padding = self.conv_layers[itr//2].padding
                stride = self.conv_layers[itr//2].stride
                if itr % 2 == 0:
                    reduced_param = im2col_indices(param, kernel_size[0], kernel_size[1], padding[0], stride[0])
                    reduced_param = reduced_param.T
                    reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param, axis=0, keepdims=True)
                else:
                    sz = param.shape
                    reduced_param = np.transpose(param,(1,0,2,3))
                    reduced_param = np.reshape(reduced_param, (sz[1],-1))
                    reduced_param = reduced_param.T
                    reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param,axis=0, keepdims=True)
            elif len(param.shape) == 2:
                reduced_param = param
            else:
                raise Exception('invalid param length = {}'.format(len(param)))
            param_out.append(reduced_param)
            #print('out param_shape = {}'.format(reduced_param.shape))
        return tuple(param_out)





