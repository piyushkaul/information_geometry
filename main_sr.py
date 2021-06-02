from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model_sr import Net
from core.data import get_training_set, get_test_set
from core.ngd import select_optimizer
from utils.arguments import argument_parser
from core.fim_model import Hook
from utils.utils import get_file_suffix, save_files
from utils.wall_clock import WallClock


parent_parser = argument_parser()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example', parents=[parent_parser], conflict_handler='resolve')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor, args=opt).to(device)
criterion = nn.MSELoss()

optimizer = select_optimizer(model, opt.optimizer, lr=opt.lr)

#if 'sgd' in opt.optimizer:
#    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
#elif 'ngd' in opt.optimizer:
#    optimizer = Adam_NGD(model.parameters(), lr=opt.lr)


def train(epoch, optimizer, train_loss_list):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        Hook.enable_all_hooks()
        output = model(input)
        if 'ngd' in opt.optimizer or opt.fim_wo_optimization:
            model.maintain_fim(opt, iteration, type_of_loss='superresolution', output=output, criterion=criterion, lr=opt.lr)
            Hook.disable_all_hooks()

        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    train_loss_list.append(epoch_loss / len(training_data_loader))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test(test_accuracy_list):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    test_accuracy_list.append(avg_psnr / len(testing_data_loader))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

train_loss_list = []
test_accuracy_list = []
elapsed_time_list = []
wall_clock = WallClock()
for epoch in range(1, opt.nEpochs + 1):
    suffix = get_file_suffix(opt)
    train(epoch, optimizer, train_loss_list)
    test(test_accuracy_list)
    elapsed_time_list.append(wall_clock.elapsed_time())
    checkpoint(epoch)

save_files(train_loss_list, 'train_loss', suffix)
save_files(test_accuracy_list, 'test_accuracy', suffix)

with open("summary_sr" + ".txt", "a") as fp_sum:
    fp_sum.writelines(['Experiment : {}\tTest Acc = {}\tTrain Loss = {}\tElapsed Time = {}\n'.format(suffix, test_accuracy_list[-1], train_loss_list[-1], elapsed_time_list[-1])])
