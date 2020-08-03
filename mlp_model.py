import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from fim_model import ModelFIM
import torch


class MLP(ModelFIM):
    def __init__(self, subspace_fraction=0.1):
        super(MLP, self).__init__()
        self.subspace_fraction = subspace_fraction
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

        self.GS = OrderedDict()
        self.GS['PSI0_AVG'] = torch.eye((784))
        self.GS['GAM0_AVG'] = torch.eye((250))
        self.GS['PSI1_AVG'] = torch.eye((250))
        self.GS['GAM1_AVG'] = torch.eye((100))
        self.GS['PSI2_AVG'] = torch.eye((100))
        self.GS['GAM2_AVG'] = torch.eye((10))
        super(MLP, self).common_init()



    def forward(self, X):
        self.a0 = X
        self.s0 = self.linear1(self.a0)
        self.a1 = F.relu(self.s0)
        self.s1 = self.linear2(self.a1)
        self.a2 = F.relu(self.s1)
        self.s2 = self.linear3(self.a2)
        self.s0.retain_grad()
        self.s1.retain_grad()
        self.s2.retain_grad()
        return F.log_softmax(self.s2, dim=1)

    def get_grads(self):
        a0 = self.a0.detach()
        s0_grad = self.s0.grad.detach()
        a1 = self.a1.detach()
        s1_grad = self.s1.grad.detach()
        a2 = self.a2.detach()
        s2_grad = self.s2.grad.detach()
        #print('a0.shape = {}, so_grad.shape = {}, a1.shape = {}, s1_grad.shape = {}, a2.shape = {}, s2_grad.shape = {}'.format(a0.shape, s0_grad.shape, a1.shape, s1_grad.shape, a2.shape, s2_grad.shape))
        return (a0, s0_grad, a1, s1_grad, a2, s2_grad)

