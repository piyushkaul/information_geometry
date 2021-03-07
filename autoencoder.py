import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from models.autoencoder_model import Autoencoder
from core.ngd import NGD
from core.ngd import select_optimizer
from torch.optim.lr_scheduler import StepLR
from utils import arguments
from utils import utils
from logger import MyLogger
import numpy as np

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
el_fim = True


def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor: tensor_round(tensor))
])

parser = arguments.argument_parser()
args = parser.parse_args()

train_dataset = MNIST('./data', transform=img_transform, download=True, train=True)
test_dataset = MNIST('./data', transform=img_transform, download=True, train=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

use_cuda = False
suffix = utils.get_file_suffix(args)
device = torch.device("cuda" if use_cuda else "cpu")
model = Autoencoder(args, init_from_rbm=True).to(device)
logger = MyLogger(train_dataloader, model, suffix=suffix)
model = Autoencoder(args, init_from_rbm=True, logger=logger).to(device)

criterion = nn.BCELoss()
optimizer = select_optimizer(model, args.optimizer, args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

loss_list = []
MSE_loss_list = []


def loss_backprop(output, img, criterion, optimizer):
    loss = criterion(output, img)
    MSE_loss = nn.MSELoss()(output, img)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    return loss, MSE_loss


def ngd_el_fim_loss(model, args, batch_idx, output, criterion, optimizer):
    loss, MSE_loss = loss_backprop(output, img, criterion, optimizer)
    model.maintain_fim(args, batch_idx)
    optimizer.step(whitening_matrices=model.GSINV)
    return loss, MSE_loss


def ngd_std_fim_loss(model, args, batch_idx, output, criterion, optimizer):
    model.maintain_fim(args, batch_idx, type_of_loss='autoencoder', output=output, criterion=criterion)
    loss, MSE_loss = loss_backprop(output, img, criterion, optimizer)
    optimizer.step(whitening_matrices=model.GSINV)
    return loss, MSE_loss


def sgd_loss(model, args, batch_idx, output, criterion, optimizer):
    loss, MSE_loss = loss_backprop(output, img, criterion, optimizer)
    if args.fim_wo_optimization:
        model.maintain_fim(args, batch_idx)
    optimizer.step()
    return loss, MSE_loss

def sgd_special_loss(model, args, batch_idx, output, criterion, optimizer):
    loss = criterion(output, img)
    def get_trace_eig(model):
        gs_values = list(model.GS.values())
        trace = 0
        for item_no, (key, item) in enumerate(model.GS.items()):
            if item_no%2==0:
                eigval_psi, eigvec_psi = torch.symeig(gs_values[item_no], eigenvectors=False)
                eigval_gam, eigvec_gam = torch.symeig(gs_values[item_no+1], eigenvectors=False)
                if model.use_cuda:
                    eigval_f_inv = np.kron(eigval_psi.cpu().numpy(), eigval_gam.cpu().numpy())
                else:
                    eigval_f_inv = np.kron(eigval_psi.numpy(), eigval_gam.numpy())
                trace = trace + np.sum(eigval_f_inv)
        return trace

    trace = get_trace_eig(model)
    MSE_loss = nn.MSELoss()(output, img)
    loss = loss + trace

    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    if args.fim_wo_optimization:
        model.maintain_fim(args, batch_idx)
    optimizer.step()
    return loss, MSE_loss



if isinstance(optimizer, NGD):
    if not el_fim:
        backprop_and_optimize = ngd_std_fim_loss
    else:
        backprop_and_optimize = ngd_el_fim_loss

else:
    backprop_and_optimize = sgd_special_loss


class LossMetric:
    def __init__(self, tag='train', logger=None):
        self.total_loss = 0
        self.total_mse = 0
        self.avg_loss = 0
        self.avg_mse = 0
        self.items = 0
        self.tag = tag
        self.logger = logger

    def update(self, loss, mse, samples):
        self.total_loss += loss.item()
        self.total_mse += mse.item()
        self.avg_loss = self.total_loss / samples
        self.avg_mse = self.total_mse / samples

    def log(self):
        self.logger.log_ae_running_loss(loss.item(), epoch, tag=self.tag)
        self.logger.log_ae_running_mse_loss(MSE_loss.item(), epoch, tag=self.tag)


for epoch in range(num_epochs):
    batch_idx = 0
    train_metric = LossMetric(tag='train', logger=logger)
    test_metric = LossMetric(tag='test', logger=logger)
    for data in train_dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = img.to(device)
        output = model(img)
        loss, MSE_loss = backprop_and_optimize(model, args, batch_idx, output, criterion, optimizer)
        batch_idx = batch_idx + 1
        train_metric.update(loss, MSE_loss, batch_idx * batch_size)

    # scheduler.step()
    # optimizer.step()
    # ===================log========================

    train_metric.log()

    print('epoch [{}/{}], loss:{}, MSE_loss:{}'
          .format(epoch + 1, num_epochs, loss.item(), MSE_loss.item()))
    loss_list.append(loss.item())
    MSE_loss_list.append(MSE_loss.item())
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))

    test_batch_idx = 0
    for data in test_dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = img.to(device)
        output = model(img)
        test_loss, test_MSE_loss = backprop_and_optimize(model, args, batch_idx, output, criterion, optimizer)
        test_batch_idx = test_batch_idx + 1
        test_metric.update(test_loss, test_MSE_loss, test_batch_idx * batch_size)

    test_metric.log()

utils.save_files(loss_list, 'loss', suffix)
utils.save_files(MSE_loss_list, 'MSE_loss', suffix)
with open("summary_autoencoder.txt", "a") as fp_sum:
    fp_sum.writelines(['Experiment : {}\tLoss = {}\tMSE Loss={}\n'.format(suffix, loss_list[-1], MSE_loss_list[-1])])

torch.save(model.state_dict(), './sim_autoencoder.pth')
