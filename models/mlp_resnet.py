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
        features = [784, 256, 256, 256, 256, 256, 256, 256, 256, 256, 10]
        self.bypass_en = True
        if self.bypass_en:
            self.linear_seq1 = self.fc_layer(features[0], features[1])
            self.linear_seq2 = self.fc_layer(features[1], features[2])
            self.linear_seq3 = self.fc_layer(features[2], features[3])
            self.linear_seq4 = self.fc_layer(features[3], features[4])
            self.linear_seq5 = self.fc_layer(features[4], features[5])
            self.linear_seq6 = self.fc_layer(features[5], features[6])
            self.linear_seq7 = self.fc_layer(features[6], features[7])
            self.linear_seq8 = self.fc_layer(features[7], features[8])
            self.linear_seq9 = self.fc_layer(features[8], features[9])
            self.linear_seq10 = nn.Linear(features[9], features[10])
        else:
            self.linear_seq = nn.Sequential(
                self.fc_layer(features[0], features[1]),
                self.fc_layer(features[1], features[2]),
                self.fc_layer(features[2], features[3]),
                self.fc_layer(features[3], features[4]),
                self.fc_layer(features[4], features[5]),
                self.fc_layer(features[5], features[6]),
                self.fc_layer(features[6], features[7]),
                self.fc_layer(features[7], features[8]),
                self.fc_layer(features[8], features[9]),
                nn.Linear(features[9], features[10]))

        #self.dropout = nn.Dropout(0.2)
        super(MLPResNet, self).common_init(args, hook_enable=hook_enable, logger=logger)

    def fc_layer(self, features_in, features_out):
        return nn.Sequential(
                             nn.Linear(features_in, features_out),
                             nn.BatchNorm1d(features_out)
                            )

    def forward(self, x):
        if self.bypass_en:
            x1 = self.linear_seq1(x)
            x2 = self.linear_seq2(x1)
            x2 = x2 + x1
            x3 = self.linear_seq3(x2)
            x4 = self.linear_seq4(x3)
            x4 = x4 + x3
            x5 = self.linear_seq5(x4)
            x6 = self.linear_seq6(x5)
            x6 = x6 + x5
            x7 = self.linear_seq7(x6)
            x8 = self.linear_seq8(x7)
            x8 = x8 + x7
            x9 = self.linear_seq9(x8)
            x = self.linear_seq10(x9)
        else:
            x = self.linear_seq(x)
        return F.log_softmax(x, dim=1)



