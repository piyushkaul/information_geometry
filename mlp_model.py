import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from fim_model import ModelFIM
import torch


class MLP(ModelFIM):
    def __init__(self, args):
        super(MLP, self).__init__(args)
        self.subspace_fraction = args.subspace_fraction
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)
        super(MLP, self).common_init(args)

    def forward(self, X):
        self.a0 = X
        self.s0 = self.linear1(self.a0)
        self.a1 = F.relu(self.s0)
        self.s1 = self.linear2(self.a1)
        self.a2 = F.relu(self.s1)
        self.s2 = self.linear3(self.a2)
        return F.log_softmax(self.s2, dim=1)



