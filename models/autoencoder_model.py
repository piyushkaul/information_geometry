import torch.nn as nn
from core.fim_model import ModelFIM
import torch

features = [784,1000,500,250,30]


class Autoencoder(ModelFIM):
    def __init__(self, args, init_from_rbm=False, hook_enable=True):
        super(Autoencoder, self).__init__(args)
        self.encoder = nn.Sequential(
            nn.Linear(features[0], features[1]),
            nn.Sigmoid(),
            nn.Linear(features[1], features[2]),
            nn.Sigmoid(),
            nn.Linear(features[2], features[3]),
            nn.Sigmoid(),
            nn.Linear(features[3], features[4]))
        self.decoder = nn.Sequential(
            nn.Linear(features[4], features[3]),
            nn.Sigmoid(),
            nn.Linear(features[3], features[2]),
            nn.Sigmoid(),
            nn.Linear(features[2], features[1]),
            nn.Sigmoid(),
            nn.Linear(features[1], features[0]),
            nn.Sigmoid())
        super(Autoencoder, self).common_init(args, hook_enable=hook_enable)
        if init_from_rbm:
            self.init_from_rbm()

    def init_from_rbm(self):
        enc_layers = [0, 2, 4, 6]
        dec_layers = [6, 4, 2, 0]
        for rbm_idx in range(4):
            file_name = 'rbm' + str(rbm_idx) + '.pt'
            loaded = torch.load(file_name)
            print('Encoder: orig_shape = {}, loaded_shape={}'.format(self.encoder[enc_layers[rbm_idx]].weight.data.shape, loaded['weights'].data.shape))
            print('Decoder: orig_shape = {}, loaded_shape={}'.format(self.decoder[dec_layers[rbm_idx]].weight.data.shape, loaded['weights'].data.shape))

            self.encoder[enc_layers[rbm_idx]].weight.data.copy_(loaded['weights'].data.T)
            self.encoder[enc_layers[rbm_idx]].bias.data.copy_(loaded['bias_fwd'].data)
            self.decoder[dec_layers[rbm_idx]].weight.data.copy_(loaded['weights'].data)
            self.decoder[dec_layers[rbm_idx]].bias.data.copy_(loaded['bias_back'].data)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

