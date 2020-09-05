import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from fim_model import ModelFIM
import torch

class Autoencoder(ModelFIM):
    def __init__(self, args):
        super(Autoencoder, self).__init__(args)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 28 * 28),
            nn.Sigmoid())
        super(Autoencoder, self).common_init(args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

