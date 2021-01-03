import torch.nn as nn
import torch.nn.functional as F
from core.fim_model import ModelFIM
import torch


class MLP(ModelFIM):
    def __init__(self, args, hook_enable=True, logger=None):
        super(MLP, self).__init__(args)
        self.subspace_fraction = args.subspace_fraction
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.linear1 = nn.Linear(784, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.linear3 = nn.Linear(2048, 10)
        #self.dropout = nn.Dropout(0.2)
        super(MLP, self).common_init(args, hook_enable=hook_enable, logger=logger)

    def forward(self, X):
        a0 = X
        s0 = self.linear1(a0)
        b0 = self.bn1(s0)
        a1 = F.relu(b0)
        s1 = self.linear2(a1)
        b1 = self.bn2(s1)
        a2 = F.relu(b1)
        #d2 = self.dropout(a2)
        d2 = a2
        s2 = self.linear3(d2)
        return F.log_softmax(s2, dim=1)



