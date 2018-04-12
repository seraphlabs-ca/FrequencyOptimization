import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import butter, lfilter, freqz
import numpy as np

#=============================================================================#
# Functions
#=============================================================================#


def butter_build_filter(cutoff, fs, order=5, btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


def butter_apply_filter(data, cutoff, fs, order=5, btype='low'):
    b, a = butter_build_filter(cutoff, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
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
        """
        f_signal_dict = {}
        for k, v in signal_dict.iteritems():
            data = self.signal_dict.get(k, [])
            if isinstance(v, float):
                d = v
            else:
                d = v.data.clone().cpu().numpy().tolist()
                if isinstance(d, list):
                    d = d[0]

            data.append(d)
            if self.active:
                f_data = butter_apply_filter(
                    data=data,
                    cutoff=self.cutoff,
                    fs=1.0,
                    order=self.order,
                    btype=self.btype,
                )
            else:
                f_data = data

            self.signal_dict[k] = data
            self.f_signal_dict[k] = self.f_signal_dict.get(k, []) + [f_data[-1]]

            if min_val is not None:
                self.f_signal_dict[k][-1] = max(min_val, self.f_signal_dict[k][-1])

            if max_val is not None:
                self.f_signal_dict[k][-1] = min(max_val, self.f_signal_dict[k][-1])

            # scale signal
            coef = self.f_signal_dict[k][-1] / self.signal_dict[k][-1] if self.signal_dict[k][-1] else 1.0

            # TODO: remove me
            coef = np.clip(coef, 1e-1, 10.0)

            f_signal_dict[k] = v * coef

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
                max_ind = np.argmax(d)
                max_val = d[max_ind]
                min_ind = np.argmin(d)
                min_val = d[min_ind]

                desc = "min = %.4e [%i] max = %.4e [%i]" % (min_val, min_ind, max_val, max_ind)

                return desc

            fig = plt.figure()
            all_figs.append(fig)
            plt.plot(data, 'k--', lw=3, label="data %s" % desc_data(data))
            plt.plot(f_data, 'r', lw=1, label="f_data %s" % desc_data(f_data))
            plt.xlabel("step")
            plt.ylabel("value")
            plt.grid()
            plt.title("%s" % (k))
            plt.legend(loc="best")
            plt.tight_layout()

        return all_figs
