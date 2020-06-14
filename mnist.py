from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.optimizer import Optimizer, required


class NGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, whitening_matrices = None, closure=None):
        """Performs a single optimization step.
        Arguments:
            whitening_matrices(optional): dictionary of whitening matrices
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            whitening_matrices_list = []
            if whitening_matrices != None:
                whitening_matrices_list_values = list(whitening_matrices.values())
                whitening_matrices_list_keys = list(whitening_matrices.keys())

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if whitening_matrices_list:
                    #print('idx = {}, whitening_matrices[{}].shape, whitening_matrices[{}].shape'.format(idx, whitening_matrices_list_keys[2*idx],whitening_matrices_list_keys[2*idx-1]))
                    psi = whitening_matrices_list_values[2 * idx - 1]
                    gamma = whitening_matrices_list_values[2 * idx ]
                    d_p = psi @ dp @gamma
                '''
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                '''
                p.add_(d_p, alpha=-group['lr'])

        return loss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)
        self.GS = OrderedDict()
        self.GS['PSI0_AVG'] = np.zeros((784, 784))
        self.GS['GAM0_AVG'] = np.zeros((250,250))
        self.GS['PSI1_AVG'] = np.zeros((250, 250))
        self.GS['GAM1_AVG'] = np.zeros((100, 100))
        self.GS['PSI2_AVG'] = np.zeros((100, 100))
        self.GS['GAM2_AVG'] = np.zeros((10, 10))
        self.GSLOWER = {}
        self.GSINV = {}
        self.P = {}
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.GSINV[key] = self.GS[key]

    def forward(self, X):
        self.a0 = X
        self.s0 = self.linear1(self.a0)
        self.a1 = F.relu(self.s0)
        self.s1 = self.linear2(self.a1)
        self.a2 = F.relu(self.s1)
        self.s2 = self.linear3(self.a2)
        self.s0.retain_grad()
        self.s1.retain_grad()
        self.s2.retain_grad()
        return F.log_softmax(self.s2, dim=1)

    '''
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(self.a1))
        self.a2 = F.relu(self.s2)
        self.s3 = self.linear3(self.a2)
        self.s1.retain_grad()
        self.s2.retain_grad()
        self.s3.retain_grad()
        return F.log_softmax(X, dim=1)
    '''

    def get_grads(self):
        a0 = self.a0.detach().numpy()
        s0_grad = self.s0.grad.detach().numpy()
        a1 = self.a1.detach().numpy()
        s1_grad = self.s1.grad.detach().numpy()
        a2 = self.a2.detach().numpy()
        s2_grad = self.s2.grad.detach().numpy()

        return (a0, s0_grad, a1, s1_grad, a2, s2_grad)

    def projection_matrix_update(self, matrix, key, percent=0.1):
        eigval, eigvec = np.linalg.eigh(matrix)
        #plt.plot(eigval)
        #plt.show()
        num_eigvals = eigval.shape[0]
        subspace_size = int(num_eigvals*percent)
        eigvec_subspace = eigvec[:,-subspace_size:]
        #print('Shape of P[{}] = {}'.format(key, eigvec_subspace.shape))
        self.P[key] = eigvec_subspace
        self.P[key] == np.eye(matrix.shape[0])

    def project_to_lower_space(self, matrix, key):
        #print('project_to_lower_space: Shape of P[{}] = {}. Shape of matrix = {}'.format(key, self.P[key].shape, matrix.shape))
        return matrix @ self.P[key]

    def project_to_higher_space(self, matrix, key):
        #print('project_to_higher_space: Shape of P[{}] = {}. Shape of matrix = {}'.format(key, self.P[key].shape, matrix.shape))
        return self.P[key] @ matrix


    def maintain_avgs(self, params):
        corr_curr = [None]*len(params)
        for item_no, item in enumerate(params):
            corr_curr[item_no] = params[item_no].T @ params[item_no]
        alpha = 0.95
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, corr_curr[item_no].shape, key, self.GS[key].shape))
            self.GS[key] = alpha * self.GS[key] + (1 - alpha) * corr_curr[item_no]

    def get_inverses(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.projection_matrix_update(self.GS[key], key)
            GSPROJ = self.project_to_lower_space(self.GS[key], key)
            GSPROJINV = np.linalg.pinv(GSPROJ)# + np.eye(GSPROJ.shape[0]) * 0.001)
            self.GSINV[key] = self.project_to_higher_space(GSPROJINV, key)


def train(args, model, device, train_loader, optimizer, epoch, cnn_model=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if not cnn_model:
            data = np.reshape(data, (data.shape[0], -1))
        #print('data size = {}, target size = {}'.format(data.shape, target.shape))
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        params = model.get_grads()
        model.maintain_avgs(params)
        if batch_idx % 50 == 0:
            model.get_inverses()

        optimizer.step(whitening_matrices=model.GSINV)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
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
    args = parser.parse_args()
    test_loss_list = []
    accuracy_loss_list = []
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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
    if args.cnn_model:
        model = Net().to(device)
    else:
        model = MLP().to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = NGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, cnn_model=args.cnn_model)
        test(model, device, test_loader,  test_loss_list, accuracy_loss_list, cnn_model=args.cnn_model)
        scheduler.step()

    plt.subplot(211)
    plt.plot(range(len(test_loss_list)), test_loss_list, 'r', label='test loss list')
    plt.subplot(212)
    plt.plot(range(len(accuracy_loss_list)), accuracy_loss_list, 'r', label='accuracy loss list')
    plt.legend(loc=2, fontsize="small")
    plt.show()
    now = datetime.now()
    date_time = now.strftime("%m%d%Y%H%M%S")

    plt.savefig('eig_vals.png')
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn" + date_time + ".pt")
    np.save('temp/losses_' + data_time +'.npy', np.array(test_loss_list))
    np.save('temp/accuracy_' + data_time + '.npy', np.array(accuracy_loss_list))

if __name__ == '__main__':
    main()