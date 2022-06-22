"""
Various transformations used for data preprocessing

RR van de Leur, MN Bos and MB Vessies
"""

import torch
import numpy as np
from scipy import interpolate


class ToTensor(object):
    """
    Converts the ECG waveform and label numpy arrays to FloatTensors.
    """
    def __call__(self, sample):
        waveform = sample['waveform']
        secondary_waveform = None
        if 'secondary_waveform' in sample:
            secondary_waveform = sample['secondary_waveform']    

        sample['waveform'] = torch.from_numpy(waveform).type(torch.float)
        if secondary_waveform is not None:
            sample['secondary_waveform'] = torch.from_numpy(secondary_waveform).type(torch.float)

        if 'label' in sample:
            sample['label'] = torch.from_numpy(
                np.array(sample['label'])).type(torch.float)

        return sample


class ApplyGain(object):
    """
    Applies the gain to the ECG signal to convert it into millivolts. The millivolt
    values of the ECG are usually in a good range for neural networks, so no
    other normalization is needed. For the physionet dataset, a correction for the 
    baseline is also performed.
    """
    def __call__(self, sample):
        sample['waveform'] = sample['waveform'] * sample['gain']
        baseline_corrections = [(int(k.split('_')[1]), v) for k, v in sample.items() if 'TrueBaseline_' in k]
        for lead_idx, correction in baseline_corrections:
            sample['waveform'][lead_idx] -= correction

        return sample


class To12Lead(object):
    """
    Convert 8 lead ECGs (with I, II, V1-V6 in this order) to their 12 lead equivalent.
    """
    def _to12lead(self, waveform):
        out = torch.zeros((12, waveform.shape[1]))
        # I and II
        out[0:2, :] = waveform[0:2, :] 
        # III = II - I
        out[2, :] = waveform[1, :] - waveform[0, :]
        # aVR = -(I + II)/2
        out[3, :] = -(waveform[0, :] + waveform[1, :]) / 2
        # aVL = I - II/2
        out[4, :] = waveform[0, :] - (waveform[1, :] / 2)
        # aVF = II - I/2
        out[5, :] = waveform[1, :] - (waveform[0, :] / 2)
        # V1 to V6
        out[6:12, :] = waveform[2:8, :]

        return out

    def __call__(self, sample):
        waveform = sample['waveform']
        
        assert waveform.shape[0] == 8, "The To12Lead transform only works with 8 channel input ECGs, please check."

        sample['waveform'] = self._to12lead(waveform)
        
        secondary_waveform = None
        if 'secondary_waveform' in sample:
            secondary_waveform = sample['secondary_waveform']
            sample['secondary_waveform'] = self._to12lead(secondary_waveform)

        return sample


class Resample(object):
    """Resamples the ECG to the specified sampling frequency.

    Attributes:
        sample_freq: The required sampling frequency to resample to.
    """
    def __init__(self, sample_freq):
        """Initializes the resample transformation.
        """
        self.sample_freq = int(sample_freq)

    def _resample(self, waveform, samplebase):
        length = int(waveform.shape[1])
        out_length = int((length / samplebase) * self.sample_freq)

        if type(waveform) == np.ndarray:
            x = np.linspace(0, length / samplebase, num=length)
            f = interpolate.interp1d(x, waveform, axis=1)
            xnew = np.linspace(0, length / samplebase, num=out_length)
            return f(xnew)
        elif type(waveform) == torch.Tensor:
            return torch.nn.functional.interpolate(waveform.unsqueeze(0), size=out_length, mode='linear', align_corners=False).squeeze(0)
                  
    def __call__(self, sample):
        samplebase = int(sample['samplebase'])
        waveform = sample['waveform']
        secondary_waveform = None
        if 'secondary_waveform' in sample:
            secondary_waveform = sample['secondary_waveform']

        if samplebase != self.sample_freq:
            sample['waveform'] = self._resample(waveform, samplebase)

            # Do the same for secondary waveform
            if secondary_waveform is not None:
                sample['secondary_waveform'] = self._resample(secondary_waveform, samplebase)

        return sample


class PolyFilter(object):
    """Filter ECG by subtracting a polynomial.

    Attributes:
        order (int): The order of the filter, usually 2, 3 or 4.
    """
    def __init__(self, order):
        """Initializes the resample transformation.
        """
        self.order = int(order)

    def __call__(self, sample):
        waveform = sample['waveform']
        ecg_out = np.zeros_like(waveform)

        for lead in range(waveform.shape[0]):
            z = np.polyfit(np.linspace(0, 10, waveform.shape[1]), waveform[lead, :], self.order)
            yp = np.poly1d(z)
            ecg_out[lead, :] = waveform[lead, :] - yp(np.linspace(0, 10, waveform.shape[1]))

        sample['waveform'] = ecg_out

        return sample
