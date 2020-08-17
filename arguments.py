import argparse

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
    parser.add_argument('--model', type=str, default='mlp',
                        help='Which NN model to use')
    parser.add_argument('--random-projection', action='store_true', default=False,
                        help='Random Projection')
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
    parser.add_argument('--grid-search', action='store_true', default=False,
                        help='Grid Search')
    parser.add_argument('--dump-eigenvalues', action='store_true', default=False,
                        help='Dump Eigenvalues')


    return parser