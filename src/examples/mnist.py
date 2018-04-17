from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import os
import traceback

from frequency_filtering import FrequencyFilter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--freq-cutoff', type=float, default=-1.0,
                    help='frequency filtering cutoff \in (0, 0.5), if < 0 then no filtering is used')
parser.add_argument('--freq-order', type=int, default=3,
                    help='frequency filtering filter order')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

freq_filter = FrequencyFilter(
    active=args.freq_cutoff > 0.0,
    cutoff=args.freq_cutoff,
    order=args.freq_order,
)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # collect unfiltered loss
        loss.backward()
        loss_dict = {
            "train.loss.val": loss,
        }

        for n, p in model.named_parameters():
            loss_dict.update({
                "train.loss.%s" % n: p.grad,
            })
        f_loss_dict = freq_filter.step(loss_dict)

        # update filtered gradients
        optimizer.zero_grad()
        for n, p in model.named_parameters():
            p.grad = Variable(torch.FloatTensor(f_loss_dict["train.loss.%s" % n]))

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    freq_filter.step({"test.loss": test_loss})
    freq_filter.step({"test.accuracy": float(correct) / len(test_loader.dataset)})


try:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
except KeyboardInterrupt as e:
    print("CTRL-C detected - stopping training")

freq_filter.plot()

# install (Common)[https://github.com/seraphlabs-ca/Common] to save plots
try:
    import common
    import common.media
    from common.root_logger import logger
    import common.options as opts

    image_path = os.path.join("..", "data", "generated", "mnist",
                              "images.freq-cutoff_%e__freq-order_%i.%s" % (
                                  args.freq_cutoff, args.freq_order,
                                  common.aux.get_fname_timestamp(),
                              ))
    logger.info("image_path = %s" % image_path)
    common.media.save_all_figs(image_path, im_type="png")
    common.media.save_all_figs(image_path, im_type="html")
    opts.Options(vars(args)).export_as_ini(os.path.join(image_path, "args"))
    # opts.Options({
    #     "signal": freq_filter.signal_dict,
    #     "f_signal": freq_filter.f_signal_dict,
    # }).export_as_json(os.path.join(image_path, "results"))
except Exception as e:
    print("Failed saving plots")
    traceback.print_exc()
