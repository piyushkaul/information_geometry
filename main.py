from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#import numpy as np
import matplotlib.pyplot as plt
from models.mlp_model import MLP
from models.cnn_model import CNN
from models.mlp_resnet import MLPResNet
from core.ngd import NGD
from core.ngd import select_optimizer#, maintain_fim
import numpy as np
from utils.utils import save_files, get_file_suffix
from utils import arguments
from models import resnet
from logger import MyLogger
import time

def train(args, model, device, train_loader, optimizer, criterion, epoch, batch_size, train_loss_list, train_accuracy_list, cnn_model=False, logger=None):
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
        total_iter = (batch_idx+1)*target.shape[0]
        total_samples = len(train_loader.dataset)

        #if isinstance(optimizer, NGD) or args.fim_wo_optimization:
        if 'ngd' in args.optimizer or args.fim_wo_optimization:
            model.maintain_fim(args, batch_idx, type_of_loss='classification', output=output, criterion=criterion)

        loss = criterion(output, target)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        if np.isnan(loss.item()):
            raise Exception('Nan in loss')

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            logger.log_train_running_loss(running_loss / (batch_idx+1)*batch_size, batch_idx*batch_size + len(train_loader.dataset)*(epoch-1))

        #if batch_idx == 10:
        #    break

    logger.log_train_loss(100. * correct / len(train_loader.dataset), running_loss/len(train_loader.dataset), epoch)
    train_loss_list.append(running_loss/len(train_loader.dataset))
    train_accuracy_list.append(100. * correct / len(train_loader.dataset))


def test(model, device, test_loader, test_loss_list, accuracy_loss_list, epoch, batch_size, cnn_model=False, logger=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            total_iter = (batch_idx+1)*target.shape[0]
            data, target = data.to(device), target.to(device)
            if not cnn_model:
                data = torch.reshape(data, (data.shape[0], -1))
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            logger.log_test_running_loss(test_loss/((batch_idx+1)*batch_size), (epoch-1)*len(test_loader.dataset) + (batch_idx*batch_size))



    test_loss /= len(test_loader.dataset)
    test_loss_list.append(test_loss)
    accuracy_loss_list.append(100. * correct / len(test_loader.dataset))
    logger.log_test_loss(100. * correct / len(test_loader.dataset), test_loss, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

def cifar10_loader(args, kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./data/CIFAR10',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def get_data_loader(args, kwargs):
    if args.dataset == 'mnist':
        train_loader, test_loader = mnist_loader(args, kwargs)
    elif args.dataset == 'fashion_mnist':
        train_loader, test_loader = fashion_mnist_loader(args, kwargs)
    elif args.dataset == 'cifar10':
        train_loader, test_loader = cifar10_loader(args, kwargs)
    else:
        raise Exception('Unknown dataset = {}'.format(args.dataset))
    return train_loader, test_loader

def get_model(args, hook_enable=True, logger=None):
    if args.model == 'cnn':
        model = CNN(args, hook_enable=hook_enable, logger=logger)
        cnn_type=True
    elif args.model == 'mlp':
        model = MLP(args, hook_enable=hook_enable, logger=logger)
        cnn_type=False
    elif args.model == 'mlp_resnet':
        model = MLPResNet(args, hook_enable=hook_enable, logger=logger)
        cnn_type=False
    elif args.model == 'resnet18':
        model = resnet.ResNet18(args, hook_enable=hook_enable, logger=logger)
        cnn_type = True
    else:
        raise Exception('Unknown Model')
    return model, cnn_type

def select_criterion(args):
    if args.model == 'cnn':
        criterion = nn.NLLLoss()
    elif args.model == 'mlp':
        criterion = nn.NLLLoss()
    elif args.model == 'resnet18':
        criterion = nn.CrossEntropyLoss()
    elif args.model == 'mlp_resnet':
        criterion = nn.NLLLoss()
    else:
        raise Exception('Unknown Model')
    return criterion

class WallClock():

    def __init__(self):
        self.start_time = time.time()

    def elapsed_time(self):
        curr_time = time.time()
        return curr_time - self.start_time

def main(args=None):
    # Training settings
    if not args:
        parser = arguments.argument_parser()
        args = parser.parse_args()
    print(args)
    suffix = get_file_suffix(args)

    test_loss_list = []
    test_accuracy_list = []
    train_loss_list = []
    train_accuracy_list = []
    elapsed_time_list = []
    if torch.cuda.is_available():
        print('Cuda is available')
    else:
        print('Cuda is not available')
        
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        print('Cuda is used')
    else:
        print('Cuda is not used')

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = get_data_loader(args, kwargs)

    model, cnn_type = get_model(args, hook_enable=False)
    logger = MyLogger(train_loader, model, suffix=suffix)
    model, cnn_type = get_model(args, hook_enable=True, logger=logger)
    model = model.to(device)



    optimizer = select_optimizer(model, args.optimizer, args.lr)
    criterion = select_criterion(args)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    wall_clock = WallClock()

    elapsed_time_list.append(wall_clock.elapsed_time()) 
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch, args.batch_size, train_loss_list, train_accuracy_list, cnn_model=cnn_type, logger=logger)
        test(model, device, test_loader,  test_loss_list, test_accuracy_list, epoch, args.batch_size,  cnn_model=cnn_type, logger=logger)
        elapsed_time_list.append(wall_clock.elapsed_time())
        scheduler.step()

        #model.epoch_bookkeeping()
        #model.dump_eigval_arrays()

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



    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn" + suffix + ".pt")

    plt.savefig('plot' + suffix + '.png')
    plt.clf()

    save_files(test_loss_list, 'test_loss', suffix)
    save_files(test_accuracy_list, 'test_accuracy', suffix)
    save_files(train_loss_list, 'train_loss', suffix)
    save_files(test_accuracy_list, 'train_accuracy', suffix)
    save_files(elapsed_time_list, 'elapsed_time', suffix)
    print('Experiment Result : {}\tTest Acc = {}\tTest Loss={}\tTrain Acc ={}\tTrain Loss = {}\tElapsed Time = {}\n'.format(suffix, test_accuracy_list[-1], test_loss_list[-1], train_accuracy_list[-1], train_loss_list[-1], elapsed_time_list[-1]))
    with open("summary_" + args.model + ".txt", "a") as fp_sum:
        fp_sum.writelines(['Experiment : {}\tTest Acc = {}\tTest Loss={}\tTrain Acc ={}\tTrain Loss = {}\tElapsed Time = {}\n'.format(suffix, test_accuracy_list[-1], test_loss_list[-1], train_accuracy_list[-1], train_loss_list[-1], elapsed_time_list[-1])])

if __name__ == '__main__':
    parser = arguments.argument_parser()
    args = parser.parse_args()
    if not args.grid_search:
        main(args)
    else:
        for model_type in ['cnn', 'mlp']:
            for gamma in [0.8, 0.7]:
                for lr in [0.1, 0.5]:
                    for opt in ['ngd', 'sgd']:#, 'sgd', 'adadelta']:
                        if opt == 'sgd':
                            args.gamma = gamma
                            args.lr = lr
                            args.optimizer = opt
                            args.subspace_fraction = 1
                            args.inv_period = 0
                            args.proj_period = 0
                            args.inv_type = 'direct'
                            if model_type == 'cnn':
                                args.cnn_model = True
                            else:
                                args.cnn_model = False
                            try:
                                main(args)
                            except:
                                print("An exception occurred")
                        else:
                            for subspace_fraction in [0.1,0.8,1]:
                                for inv_period in [50]:
                                    for proj_period in [50]:
                                        for inv_type in ['direct', 'recursive']:
                                            if model_type == 'cnn':
                                                args.cnn_model = True
                                            else:
                                                args.cnn_model = False
                                            args.gamma = gamma
                                            args.lr = lr
                                            args.optimizer = opt
                                            args.subspace_fraction = subspace_fraction
                                            args.inv_period = inv_period
                                            args.proj_period = proj_period
                                            args.inv_type = inv_type
                                            try:
                                                main(args)
                                            except:
                                                print("An exception occurred")

