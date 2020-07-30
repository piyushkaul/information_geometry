from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.optimizer import Optimizer, required
from mlp_model import MLP
from cnn_model import CNN
from ngd import NGD
from ngd import select_optimizer
import numpy as np

def train(args, model, device, train_loader, optimizer, epoch, train_loss_list, train_accuracy_list, cnn_model=False):
    model.train()
    running_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if not cnn_model:
            data = torch.reshape(data, (data.shape[0], -1))
        #print('data size = {}, target size = {}'.format(data.shape, target.shape))
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = F.nll_loss(output, target)
        loss.backward()
        running_loss += loss.item()

        if isinstance(optimizer, NGD):
            #nn.utils.clip_grad_norm_(model.parameters(), 0.000001),
            params = model.get_grads()
            model.maintain_invs(params, args)
            if batch_idx % args.proj_period == 0:
                model.projection_matrix_update()
            optimizer.step(whitening_matrices=model.GSINV)
        else:
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss_list.append(running_loss/len(train_loader.dataset))
    train_accuracy_list.append(100. * correct / len(train_loader.dataset))


def test(model, device, test_loader, test_loss_list, accuracy_loss_list, cnn_model=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if not cnn_model:
                data = np.reshape(data, (data.shape[0], -1))
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss_list.append(test_loss)
    accuracy_loss_list.append(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_files(loss_list, tag, suffix):
    loss_list_np = np.array(loss_list)
    loss_filename = 'temp/' + tag + suffix + 'txt'
    loss_list_np.tofile(loss_filename, '\n', '%f')

def argument_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--cnn-model', action='store_true', default=False,
                        help='Use CNN model now')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Optimizer to Use [sgd|adadelta|ngd]')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset to Use. [mnist|fashion_mnist]')
    parser.add_argument('--subspace-fraction', type=float, default=0.1,
                        help='Fraction of Subspace to use for NGD 0 < frac < 1')
    parser.add_argument('--inv-period', type=int, default=50,
                        help='batches after which inverse is calculated')
    parser.add_argument('--inv-type', type=str, default='direct',
                        help='method of calculation of inverse')
    parser.add_argument('--proj-period', type=int, default=50,
                        help='batches after which projection is taken')


    return parser

def mnist_loader(args, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

def fashion_mnist_loader(args, kwargs):
    normalize = transforms.Normalize(mean=[0.5],
                                     std=[0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            root='./data/FashionMNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            root='./data/FashionMNIST',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size = args.batch_size, shuffle = True, ** kwargs)

    return train_loader, test_loader

def get_data_loader(args, kwargs):
    if args.dataset == 'mnist':
        train_loader, test_loader = mnist_loader(args, kwargs)
    elif args.dataset == 'fashion_mnist':
        train_loader, test_loader = fashion_mnist_loader(args, kwargs)
    else:
        raise Exception('Unknown dataset = {}'.format(args.dataset))
    return train_loader, test_loader

def main(args=None):
    # Training settings
    if not args:
        parser = argument_parser()
        args = parser.parse_args()
    test_loss_list = []
    test_accuracy_list = []
    train_loss_list = []
    train_accuracy_list = []
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.cnn_model:
        model = CNN(subspace_fraction=args.subspace_fraction).to(device)
    else:
        model = MLP(subspace_fraction=args.subspace_fraction).to(device)

    train_loader, test_loader = get_data_loader(args, kwargs)
    optimizer = select_optimizer(model, args.optimizer, args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_loss_list, train_accuracy_list, cnn_model=args.cnn_model)
        test(model, device, test_loader,  test_loss_list, test_accuracy_list, cnn_model=args.cnn_model)
        scheduler.step()

    plt.subplot(211)
    plt.plot(range(len(test_loss_list)), test_loss_list, 'r', label='test loss')
    plt.plot(range(len(train_loss_list)), train_loss_list, 'b', label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=2, fontsize="small")

    plt.subplot(212)
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list, 'r', label='test accuracy')
    plt.plot(range(len(train_accuracy_list)), train_accuracy_list, 'b', label='train accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=2, fontsize="small")

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    suffix = date_time + '_' + args.optimizer + '_lr_' + str(args.lr) + '_gamma_' + str(args.gamma) + '_frac_' + \
             str(args.subspace_fraction) + '_' + args.dataset + '_' + args.inv_type + '_inv_period_' + str(args.inv_period)\
             + '_proj_period_' + str(args.proj_period)



    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn" + suffix + ".pt")


    plt.savefig('plot' + suffix + '.png')
    plt.clf()

    save_files(test_loss_list, 'test_loss', suffix)
    save_files(test_accuracy_list, 'test_accuracy', suffix)
    save_files(train_loss_list, 'train_loss', suffix)
    save_files(test_accuracy_list, 'train_accuracy', suffix)
    with open("summary.txt", "a") as fp_sum:
        fp_sum.writelines(['Experiment : {}\tTest Acc = {}\tTest Loss={}\tTrain Acc ={}\tTrain Loss = {}\n'.format(suffix, test_accuracy_list[-1], test_loss_list[-1], train_accuracy_list[-1], train_loss_list[-1])])

if __name__ == '__main__':
    if True:
        main()
    else:
        parser = argument_parser()
        args = parser.parse_args()
        for gamma in [0.5,0.6,0.7,0.8,0.9]:
            for lr in [0.5,0.4,0.2,0.1]:
                for opt in ['ngd']:#, 'sgd', 'adadelta']:
                    for subspace_fraction in [0.1,0.2,0.4,0.8]:
                        for inv_period in [10,50,100]:
                            for proj_period in [10,50,100]:
                                for inv_type in ['direct', 'recursive']:
                                    args.gamma = gamma
                                    args.lr = lr
                                    args.optimizer = opt
                                    args.subspace_fraction = subspace_fraction
                                    args.inv_period = inv_period
                                    args.proj_period = proj_period
                                    args.inv_type = inv_type
                                    main(args)
