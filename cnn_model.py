from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from fim_model import ModelFIM

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
        self.conv_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.GS = OrderedDict()
        self.GS['PSI0_AVG'] = torch.eye((1*3*3))
        self.GS['GAM0_AVG'] = torch.eye((32))
        self.GS['PSI1_AVG'] = torch.eye((32*3*3))
        self.GS['GAM1_AVG'] = torch.eye((64))
        self.GS['PSI2_AVG'] = torch.eye((1600))
        self.GS['GAM2_AVG'] = torch.eye((128))
        self.GS['PSI3_AVG'] = torch.eye((128))
        self.GS['GAM3_AVG'] = torch.eye((10))
        super(CNN, self).common_init(args)
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
        self.s0.retain_grad()
        self.s1.retain_grad()
        self.s2.retain_grad()
        self.s3.retain_grad()

        return output


    def get_grads(self):
        a0 = self.a0.detach()
        s0_grad = self.s0.grad.detach()
        a1 = self.a1.detach()
        s1_grad = self.s1.grad.detach()
        a2 = self.a2.detach()
        s2_grad = self.s2.grad.detach()
        a3 = self.a3.detach()
        s3_grad = self.s3.grad.detach()
        # print('a0.shape = {}, so_grad.shape = {}, a1.shape = {}, s1_grad.shape = {}, a2.shape = {}, s2_grad.shape = {}'.format(a0.shape, s0_grad.shape, a1.shape, s1_grad.shape, a2.shape, s2_grad.shape))
        params = [a0, s0_grad, a1, s1_grad, a2, s2_grad, a3, s3_grad]
        param_out = []
        for itr, (key, val) in enumerate(self.GS.items()):
            param = params[itr]
            if len(param.shape) == 4 :
                kernel_size = self.conv_layers[itr//2].kernel_size
                padding = self.conv_layers[itr//2].padding
                stride = self.conv_layers[itr//2].stride
                if itr % 2 == 0:
                    #reduced_param = im2col_indices(param, kernel_size[0], kernel_size[1], padding[0], stride[0])
                    reduced_param = torch.nn.functional.unfold(param, (kernel_size[0], kernel_size[1]), stride=1)
                    reduced_param = reduced_param.transpose(1,2)
                    reduced_param = reduced_param.reshape(-1, reduced_param.shape[-1])
                    #reduced_param = reduced_param.T
                    self.spatial_sizes[key] = 1.0/(param.shape[2] * param.shape[3])
                    self.batch_sizes[key] = param.shape[0]
                    #reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param, axis=0, keepdims=True)
                    num_samples = min(reduced_param.shape[0], 200)
                    random_samples = torch.randint(0, reduced_param.shape[0], (num_samples,))
                    reduced_param =  reduced_param[random_samples, :]
                else:
                    sz = param.shape
                    reduced_param = param.permute(1,0,2,3)
                    reduced_param = torch.reshape(reduced_param, (sz[1],-1))
                    reduced_param = reduced_param.T
                    self.spatial_sizes[key] = (param.shape[2] * param.shape[3])
                    self.batch_sizes[key] = reduced_param.shape[0]
                    #reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param,axis=0, keepdims=True)
                    num_samples = min(reduced_param.shape[0], 200)
                    random_samples = torch.randint(0, reduced_param.shape[0], (num_samples,))
                    reduced_param =  reduced_param[random_samples, :]

            elif len(param.shape) == 2:
                reduced_param = param
                self.spatial_sizes[key] = 1
                self.batch_sizes[key] = param.shape[0]
            else:
                raise Exception('invalid param length = {}'.format(len(param)))
            param_out.append(reduced_param)
            #print('out param_shape = {}'.format(reduced_param.shape))
        return tuple(param_out)





