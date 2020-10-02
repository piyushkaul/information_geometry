import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from fim_model import ModelFIM
import torch

#784,1000,500,250,3

class Autoencoder(ModelFIM):
    def __init__(self, args):
        super(Autoencoder, self).__init__(args)
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 500),
            nn.Sigmoid(),
            nn.Linear(500, 250),
            nn.Sigmoid(),
            nn.Linear(250, 30))
        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            nn.Sigmoid(),
            nn.Linear(250, 500),
            nn.Sigmoid(),
            nn.Linear(500, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 28 * 28),
            nn.Sigmoid())
        super(Autoencoder, self).common_init(args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

