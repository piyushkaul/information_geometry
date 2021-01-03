import torch.nn as nn
import torch.nn.functional as F
from core.fim_model import ModelFIM
import torch


class MLPResNet(ModelFIM):
    def __init__(self, args, hook_enable=True, logger=None):
        super(MLPResNet, self).__init__(args)
        self.subspace_fraction = args.subspace_fraction
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        features = [784, 256, 256, 256, 256, 1024, 1024, 1024, 1024, 2048, 10]

        self.linear_seq = nn.Sequential(
            self.fc_layer(features[0], features[1]),
            self.fc_layer(features[1], features[2]),
            self.fc_layer(features[2], features[3]),
            self.fc_layer(features[3], features[4)),
            self.fc_layer(features[4], features[5]),
            self.fc_layer(features[5], features[6]),
            self.fc_layer(features[5], features[6]),
            self.fc_layer(features[5], features[6]),
            self.fc_layer(features[5], features[6]),
            self.fc_layer(features[0], features[1]),
            self.nn.Linear(2048, 10))

        #self.dropout = nn.Dropout(0.2)
        super(MLP, self).common_init(args, hook_enable=hook_enable, logger=logger)

    def fc_layer(features_in, features_out):
            return nn.Sequential(nn.Linear(features_in, features_out), nn.BatchNorm1d(features_out))


    def forward(self, x):
        s2 = self.linear_seq(x)
        return F.log_softmax(s2, dim=1)



