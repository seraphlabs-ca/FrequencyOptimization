import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import butter, filtfilt, freqz
import numpy as np

import torch
from torch.autograd import Variable
#=============================================================================#
# Functions
#=============================================================================#

# TODO: test filtfilt
# TODO: use filter for FFT


def butter_build_filter(cutoff, fs, order=5, btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


def butter_apply_filter(data, cutoff, fs, order=5, btype='low'):
    b, a = butter_build_filter(cutoff, fs, order=order, btype=btype)
    y = filtfilt(b, a, data, method="gust")
    return y


class FrequencyFilter(object):
    """
    Return a low-pass filtered value to pytoch variables

    It implements a weighting scheme (for loss and gradients) of filtered(loss) / loss with the following behaviour:
        loss > filtered(loss) => reduce weight
        loss < filtered(loss) => increase weight
        which give higher weights when current loss better then current estimate.
    """

    def __init__(self, active=True, cutoff=0.1, order=3, btype='low'):
        self.active = active
        self.cutoff = cutoff
        self.order = order
        self.btype = btype

        self.signal_dict = {}
        self.f_signal_dict = {}

    def step(self, signal_dict, min_val=None, max_val=None):
        """
        Returns a filtered version of signal_dict

        signal_dict - a dictionary with scalar of current value.

        min_val/max_val - min/max coefficient values
        """
        f_signal_dict = {}
        for k, v in signal_dict.iteritems():
            data = self.signal_dict.get(k, [])
            if isinstance(v, torch.Tensor) or isinstance(v, torch.Variable):
                # d = v.clone().detach().cpu().numpy().tolist()
                d = v.data.clone().cpu().numpy().tolist()
            else:
                d = v

            data.append(d)
            if self.active and (self.cutoff > 0.0) and (self.cutoff < 0.5):
                f_data = butter_apply_filter(
                    data=np.reshape(data, (len(data), -1)),
                    cutoff=self.cutoff,
                    fs=1.0,
                    order=self.order,
                    btype=self.btype,
                )
                f_data = np.reshape(f_data, (len(data),) + tuple(np.shape(d)))
            else:
                f_data = data

            self.signal_dict[k] = data
            self.f_signal_dict[k] = self.f_signal_dict.get(k, []) + [f_data[-1]]

            # # scale signal
            # coef = self.f_signal_dict[k][-1] / self.signal_dict[k][-1] if self.signal_dict[k][-1] else 1.0

            # # limit coef
            # if max_val is not None:
            #     coef = np.clip(coef, None, max_val)
            # if min_val is not None:
            #     coef = np.clip(coef, min_val, None)

            # f_signal_dict[k] = v * coef

            f_signal_dict[k] = np.copy(f_data[-1])

        return f_signal_dict

    def plot(self):
        """
        Plots all stored data
        """
        all_figs = []
        for k in self.signal_dict.keys():
            data = self.signal_dict[k]
            f_data = self.f_signal_dict[k]

            def desc_data(d):
                try:
                    max_ind = np.argmax(d)
                    max_val = d[max_ind]
                    min_ind = np.argmin(d)
                    min_val = d[min_ind]

                    desc = "min = %.4e [%i] max = %.4e [%i]" % (min_val, min_ind, max_val, max_ind)
                except:
                    desc = ""

                return desc
            try:
                fig = plt.figure()
                plt.plot(data, 'k--', lw=3, label="data %s" % desc_data(data))
                plt.plot(f_data, 'r', lw=1, label="f_data %s" % desc_data(f_data))
                plt.xlabel("step")
                plt.ylabel("value")
                plt.grid()
                plt.title("%s" % (k))
                plt.legend(loc="best")
                plt.tight_layout()
            except:
                plt.close(fig)
            else:
                all_figs.append(fig)

        return all_figs


class AccurateFrequencyFilter(FrequencyFilter):
    """
    Return a low-pass filtered value + gradients to pytoch variables
    """

    def __init__(self, active=True, cutoff=0.1, order=3, btype='low'):
        super(AccurateFrequencyFilter, self).__init__(
            active=active,
            cutoff=cutoff,
            order=order,
            btype=btype,
        )

        # build filter coefs
        b, a = butter_build_filter(
            cutoff=cutoff,
            fs=fs,
            order=order,
            btype=btype,
        )

        self.b = Variable(b)
        self.M = len(b)
        self.a = Variable(a)
        self.N = len(a)

        if self.M != self.N:
            raise ValueError("filter len(a) != len(b)")

        # initial value is collected
        self.x = []
        self.y = []

    def step(self, signal_dict, min_val=None, max_val=None):
        """
        Apply butterworth filter
        """
        # TODO: FINISH ME - clone variable + grad (retain_grad()), return weighted value + weighted grad
        f_signal_dict = {}
        for k, v in signal_dict.iteritems():
            x = self.sig
            if len(self.x) < self.M:
                self.x.append(x)
                self.y.append(x * 0.0)
            self.x = np.hstack((x, self.x[:-1]))
            y = sum(self.b * self.x[:len(self.b)])
            y -= sum(self.a[1:] * self.y[:len(self.a) - 1])
            y /= self.a[0]
            self.y = np.hstack((y, self.y[:-1]))

        return y
