from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import os
import traceback
from frequency_filtering import FrequencyFilter


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
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

try:
    import common
    import common.media
    from common.root_logger import logger
    import common.options as opts

    image_path = os.path.join("..", "data", "generated", "vae",
                              "images.freq-cutoff_%e__freq-order_%i.%s" % (
                                  args.freq_cutoff, args.freq_order,
                                  common.aux.get_fname_timestamp(),
                              ))
    logger.info("image_path = %s" % image_path)
except:
    image_path = "./"

os.system("mkdir -p %s/results" % image_path)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

freq_filter = FrequencyFilter(
    active=args.freq_cutoff > 0.0,
    cutoff=args.freq_cutoff,
    order=args.freq_order,
)

# Reconstruction + KL divergence losses summed over all elements and batch


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss = freq_filter.step({"train.loss": loss})["train.loss"]
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       '%s/results/reconstruction_' % image_path + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    freq_filter.step({"test.loss": test_loss})


try:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        sample = Variable(torch.randn(64, 20))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
                   '%s/results/sample_' % image_path + str(epoch) + '.png')
except KeyboardInterrupt as e:
    print("CTRL-C detected - stopping training")

freq_filter.plot()

# install (Common)[https://github.com/seraphlabs-ca/Common] to save plots
try:
    common.media.save_all_figs(image_path, im_type="png")
    common.media.save_all_figs(image_path, im_type="html")
    opts.Options(vars(args)).export_as_ini(os.path.join(image_path, "args"))
    opts.Options({
        "signal": freq_filter.signal_dict,
        "f_signal": freq_filter.f_signal_dict,
    }).export_as_json(os.path.join(image_path, "results"))
except Exception as e:
    print("Failed saving plots")
    traceback.print_exc()
