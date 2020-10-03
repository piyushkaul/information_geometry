import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from autoencoder_model import Autoencoder
from ngd import NGD
from ngd import select_optimizer
from torch.optim.lr_scheduler import StepLR
import arguments
import utils

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
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    transforms.Lambda(lambda tensor:tensor_round(tensor))
])

parser = arguments.argument_parser()
args = parser.parse_args()

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
model = Autoencoder(args, init_from_rbm=True).to(device)

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
    model.maintain_fim(args, batch_idx, type_of_loss=True, output=output, criterion=criterion)
    loss, MSE_loss = loss_backprop(output, img, criterion, optimizer)
    optimizer.step(whitening_matrices=model.GSINV)
    return loss, MSE_Loss

def sgd_loss(model, args, batch_idx, output, criterion, optimizer):
    loss, MSE_loss = loss_backprop(output, img, criterion, optimizer)
    optimizer.step()
    return loss, MSE_loss

if isinstance(optimizer, NGD):
    if not el_fim:
        backprop_and_optimize = ngd_std_fim_loss
    else:
        backprop_and_optimize = ngd_el_fim_loss
              
else:
    backprop_and_optimize = sgd_loss
 

for epoch in range(num_epochs):
    batch_idx = 0
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = img.to(device)
        output = model(img)
        loss, MSE_loss = backprop_and_optimize(model, args, batch_idx, output, criterion, optimizer)
        batch_idx = batch_idx + 1

    #scheduler.step()
        #optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{}, MSE_loss:{}'
          .format(epoch + 1, num_epochs, loss.item(), MSE_loss.item()))
    loss_list.append(loss.item())
    MSE_loss_list.append(MSE_loss.item())
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))

suffix = utils.get_file_suffix(args)
utils.save_files(loss_list, 'loss', suffix)
utils.save_files(MSE_loss_list, 'MSE_loss', suffix)
with open("summary_autoencoder.txt", "a") as fp_sum:
    fp_sum.writelines(['Experiment : {}\tLoss = {}\tMSE Loss={}\n'.format(suffix, loss_list[-1], MSE_loss_list[-1])])

torch.save(model.state_dict(), './sim_autoencoder.pth')
