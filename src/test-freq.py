from common import options as opts

import torch
from torch.autograd import Variable
import numpy as np
import frequency_filtering as ff

import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import butter, filtfilt, freqz
import numpy as np


lin_gt = torch.nn.Linear(5, 1)
lin = torch.nn.Linear(5, 1)
criterion = torch.nn.MSELoss()

params = dict(lin.named_parameters())

active = True
cutoff = 0.1

filt = ff.FrequencyFilter(active=active, cutoff=cutoff)
optimizer = torch.optim.Adam(lin.parameters(), lr=1e-1)

all_params = opts.Options()

for i in range(500):
    # zero gradients
    optimizer.zero_grad()

    # create training data
    x = torch.randn((10, 5))
    pred = lin(x)
    y = lin_gt(x).detach()
    # y = y + torch.randn_like(y) * 0.01
    loss = criterion(pred, y)

    # collect current gradients
    loss.backward()

    # filt.step({
    #     "loss": loss.data.clone().cpu().numpy().astype(np.float32),
    # })

    # # collect filtered gradients
    # filt_grad = filt.step({k: p.grad.data.clone().cpu().numpy().astype(np.float32) for k, p in params.iteritems()})

    all_params *= {
        "loss": loss.data.clone().cpu().numpy().astype(np.float32).tolist(),
    }
    all_params *= {k: np.squeeze(p.grad.data.clone().cpu().numpy().astype(np.float32)).tolist()
                   for k, p in params.iteritems()}
    filt_params = all_params.map(lambda x: ff.butter_apply_filter(x, cutoff, 1.0, btype='high'))
    if active:
        optimizer.zero_grad()
        # store filter gradients
        for k, p in params.iteritems():
            if p.grad is not None:
                p.grad += Variable(torch.FloatTensor(
                    np.array(filt_params[k][-1], dtype=np.float32).reshape(tuple(p.grad.shape),)))

    optimizer.step()
    print i, loss.detach().cpu().numpy()

gt_params = opts.Options(dict(lin_gt.named_parameters()))
pred_params = opts.Options(dict(lin.named_parameters()))

print "\nGT\n", gt_params
print "\nPRED\n", pred_params
print "\nERR\n", pred_params.apply(lambda k, v: torch.abs(v - gt_params.retrieve(k)))

# filt.plot()

for k, v in all_params.iteritems():
    v = v
    filt_v = filt_params[k]
    plt.figure()
    plt.plot(v, label="v")
    plt.plot(filt_v, label="filt_v")
    plt.legend(loc='best')
    plt.grid()
    plt.title(k)
