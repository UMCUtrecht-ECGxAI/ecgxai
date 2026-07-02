"""
Various transformations used for data preprocessing

Tim Paquaij
"""

import torch
import numpy as np
from numpy.polynomial import Polynomial
from scipy import interpolate, signal
import warnings
from typing import List, Optional, Dict, Union, Tuple, Any
import contextlib

warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")


class VerifyNumberofLeads(object):
    """
    Resamples the ECG to the specified sampling frequency.

    Attributes:
        sample_freq: The required sampling frequency to resample to.
    """

    def __init__(self, num_leads):
        """Initializes the resample transformation."""
        self.num_leads = int(num_leads)

    def __call__(self, sample):
        assert (
            sample["waveform"].shape[0] == self.num_leads
        ), f"Incorrect number of leads for sample with id: {sample['id']}"

        return sample


class ToTensor(object):
    """
    Converts the ECG waveform and label numpy arrays to FloatTensors.
    """

    def __call__(self, sample):
        waveform = sample["waveform"]
        secondary_waveform = None
        if "secondary_waveform" in sample:
            secondary_waveform = sample["secondary_waveform"]

        sample["waveform"] = torch.from_numpy(waveform).type(torch.float)
        if secondary_waveform is not None:
            sample["secondary_waveform"] = torch.from_numpy(secondary_waveform).type(
                torch.float
            )

        if "median_beat" in sample:
            sample["median_beat"] = torch.from_numpy(sample["median_beat"]).type(torch.float)

        if "label" in sample:
            sample["label"] = torch.from_numpy(np.array(sample["label"])).type(
                torch.float
            )

        return sample

class ReplaceNaN(object):
    """
    Replaces NaN values with 0 in the median beat waveform.
    Should be applied before ToTensor.
    """

    def __call__(self, sample):

        if "waveform" in sample:
            sample["waveform"] = np.nan_to_num(sample["waveform"], nan=0.0)

        if "median_beat" in sample:
            sample["median_beat"] = np.nan_to_num(sample["median_beat"], nan=0.0)

        return sample


class ApplyGain(object):
    """
    Applies gain and optional baseline correction.

    Supports:
        [leads, samples]
        [segments, leads, samples]
    """
    def __init__(self, uV: bool = False):
        self.uV = 1000 if uV else 1

    def __call__(self, sample):

        wf = sample["waveform"]
        original_ndim = wf.ndim

        if wf.ndim == 2:
            wf = wf[None, ...]  # [1, leads, samples]

        if wf.ndim != 3:
            raise ValueError("Waveform must be 2D or 3D")

        # --- Apply gain to primary waveform ---
        wf = wf * sample["gain"] * self.uV

        # --- Apply per-lead baseline corrections ---
        baseline_corrections = {
            int(k.split("_")[1]): v
            for k, v in sample.items()
            if k.startswith("TrueBaseline_")
        }

        for lead_idx, correction in baseline_corrections.items():
            wf[:, lead_idx, :] -= correction

        sample["waveform"] = wf[0] if original_ndim == 2 else wf

        # --- Secondary waveform amplitude check ---
        # if "secondary_waveform" in sample:
        #     sec = sample["secondary_waveform"]
        #     sec_ndim = sec.ndim

        #     if sec.ndim == 2:
        #         sec = sec[None, ...]

        #     if sec.ndim != 3:
        #         raise ValueError("secondary_waveform must be 2D or 3D")

        #     # amplitude ranges (global, robust)
        #     ref_range = np.nanmax(wf) - np.nanmin(wf)
        #     sec_range = np.nanmax(sec) - np.nanmin(sec)

        #     if ref_range > 0:
        #         range_ratio = sec_range / ref_range

        #         tol_low, tol_high = 0.1, 10.0

        #         if not (tol_low <= range_ratio <= tol_high):
        #             sec = sec * sample["gain"]

        #     sample["secondary_waveform"] = sec[0] if sec_ndim == 2 else sec

        return sample


class To12Lead(object):
    """
    Convert 8-lead ECGs (I, II, V1–V6) to 12-lead ECGs.
    """

    def __init__(self):
        self._warned_primary_12 = False
        self._warned_secondary_12 = False

    def _to12lead(self, waveform):
        out = np.zeros((12, waveform.shape[1]))

        # I and II
        out[0:2, :] = waveform[0:2, :]

        # Derived limb leads
        out[2, :] = waveform[1, :] - waveform[0, :]  # III
        out[3, :] = -(waveform[0, :] + waveform[1, :]) / 2  # aVR
        out[4, :] = waveform[0, :] - waveform[1, :] / 2  # aVL
        out[5, :] = waveform[1, :] - waveform[0, :] / 2  # aVF

        # Precordial leads
        out[6:12, :] = waveform[2:8, :]

        return out

    def _handle_waveform(self, waveform, warn_flag, label, sample_id=None):
        num_leads = waveform.shape[0]

        if num_leads == 8:
            return self._to12lead(waveform), warn_flag

        if num_leads == 12:
            if not warn_flag:
                prefix = f"{sample_id}: " if sample_id is not None else ""
                print(f"{prefix}{label} already 12 leads, skipping To12Lead transform.")
                warn_flag = True
            return waveform, warn_flag

        raise AssertionError(
            f"The To12Lead transform only works with 8-channel input ECGs, "
            f"loaded {label} has {num_leads} leads."
        )

    def __call__(self, sample):
        # --- Primary waveform
        sample["waveform"], self._warned_primary_12 = self._handle_waveform(
            sample["waveform"],
            self._warned_primary_12,
            label="waveform",
            sample_id=sample.get("id"),
        )
        if "secondary_waveform" in sample:
            sample["secondary_waveform"], self._warned_secondary_12 = (
                self._handle_waveform(
                    sample["secondary_waveform"],
                    self._warned_secondary_12,
                    label="secondary_waveform",
                    sample_id=sample.get("id"),
                )
            )

        return sample

class To8Lead(object):
    """
    Convert 12-lead ECGs to 8-lead ECGs (I, II, V1–V6).
    """

    def __init__(self):
        self._warned_primary_8 = False
        self._warned_secondary_8 = False
        self._warned_median_8 = False
        self._warned_secondary_median_8 = False

    def _to8lead(self, waveform):
        out = np.zeros((8, waveform.shape[1]))
        out[0:2, :] = waveform[0:2, :]   # I, II
        out[2:8, :] = waveform[6:12, :]  # V1–V6
        return out

    def _handle_waveform(self, waveform, warn_flag, label, sample_id=None):
        num_leads = waveform.shape[0]

        if num_leads == 12:
            return self._to8lead(waveform), warn_flag

        if num_leads == 8:
            if not warn_flag:
                prefix = f"{sample_id}: " if sample_id is not None else ""
                print(f"{prefix}{label} already 8 leads, skipping To8Lead transform.")
                warn_flag = True
            return waveform, warn_flag

        raise AssertionError(
            f"The To8Lead transform only works with 12-channel input ECGs, "
            f"loaded {label} has {num_leads} leads."
        )

    def __call__(self, sample):
        # --- Primary waveform
        sample["waveform"], self._warned_primary_8 = self._handle_waveform(
            sample["waveform"],
            self._warned_primary_8,
            label="waveform",
            sample_id=sample.get("id"),
        )

        # --- Secondary waveform
        if "secondary_waveform" in sample:
            sample["secondary_waveform"], self._warned_secondary_8 = self._handle_waveform(
                sample["secondary_waveform"],
                self._warned_secondary_8,
                label="secondary_waveform",
                sample_id=sample.get("id"),
            )

        # --- Median beat
        if "median_beat" in sample:
            sample["median_beat"], self._warned_median_8 = self._handle_waveform(
                sample["median_beat"],
                self._warned_median_8,
                label="median_beat",
                sample_id=sample.get("id"),
            )

        # --- Secondary median beat
        if "secondary_median_beat" in sample:
            sample["secondary_median_beat"], self._warned_secondary_median_8 = self._handle_waveform(
                sample["secondary_median_beat"],
                self._warned_secondary_median_8,
                label="secondary_median_beat",
                sample_id=sample.get("id"),
            )

        return sample

class SelectLead:
    """
    Select a subset of ECG leads from a waveform.

    Parameters
    ----------
    available_leads : List[str]
        Ordered list of leads present in the waveform.
    selected_leads : List[str]
        Leads to select from the waveform.
    """

    def __init__(
        self,
        available_leads: List[str],
        selected_leads: List[str],
    ) -> None:
        if not available_leads:
            raise ValueError("`available_leads` cannot be empty.")

        if not selected_leads:
            raise ValueError("`selected_leads` cannot be empty.")

        self.available_leads = available_leads
        self.selected_leads = selected_leads

        missing = [
            lead for lead in selected_leads if lead not in available_leads
        ]
        if missing:
            raise ValueError(
                f"Selected leads not available: {missing}. "
                f"Available leads: {available_leads}"
            )

        self._lead_indices = [
            available_leads.index(lead) for lead in selected_leads
        ]

    def _select(self, waveform: np.ndarray, label: str) -> np.ndarray:
        if waveform.ndim != 2:
            raise ValueError(
                f"{label} must be 2D (leads, time). "
                f"Got shape {waveform.shape}"
            )

        if waveform.shape[0] != len(self.available_leads):
            raise ValueError(
                f"{label} lead dimension mismatch. "
                f"Expected {len(self.available_leads)}, "
                f"got {waveform.shape[0]}"
            )

        return waveform[self._lead_indices, :]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["waveform"] = self._select(
            sample["waveform"],
            label="waveform",
        )

        if "secondary_waveform" in sample:
            sample["secondary_waveform"] = self._select(
                sample["secondary_waveform"],
                label="secondary_waveform",
            )

        return sample

class To1Lead(object):
    """
    Convert 8 lead ECGs (with I, II, V1-V6 in this order) to their 1 lead equivalent.
    """

    def _to1lead(self, waveform):
        out = np.zeros((1, waveform.shape[1]))
        out[0, :] = waveform[0, :]

        return out

    def __call__(self, sample):
        waveform = sample["waveform"]

        if waveform.shape[0] != 8:
            print(sample["id"], waveform.shape)

        assert (
            waveform.shape[0] == 8
        ), "The To1Lead transform only works with 8 channel input ECGs, please check."

        sample["waveform"] = self._to1lead(waveform)

        secondary_waveform = None
        if "secondary_waveform" in sample:
            secondary_waveform = sample["secondary_waveform"]
            sample["secondary_waveform"] = self._to1lead(secondary_waveform)

        return sample


class To8Lead(object):
    """
    Convert 12-lead ECGs to 8-lead ECGs (I, II, V1–V6).
    """

    def __init__(self):
        self._warned_primary_8 = False
        self._warned_secondary_8 = False
        self._warned_median_8 = False
        self._warned_secondary_median_8 = False

    def _to8lead(self, waveform):
        out = np.zeros((8, waveform.shape[1]))
        out[0:2, :] = waveform[0:2, :]   # I, II
        out[2:8, :] = waveform[6:12, :]  # V1–V6
        return out

    def _handle_waveform(self, waveform, warn_flag, label, sample_id=None):
        num_leads = waveform.shape[0]

        if num_leads == 12:
            return self._to8lead(waveform), warn_flag

        if num_leads == 8:
            if not warn_flag:
                prefix = f"{sample_id}: " if sample_id is not None else ""
                print(f"{prefix}{label} already 8 leads, skipping To8Lead transform.")
                warn_flag = True
            return waveform, warn_flag

        raise AssertionError(
            f"The To8Lead transform only works with 12-channel input ECGs, "
            f"loaded {label} has {num_leads} leads."
        )

    def __call__(self, sample):
        # --- Primary waveform
        sample["waveform"], self._warned_primary_8 = self._handle_waveform(
            sample["waveform"],
            self._warned_primary_8,
            label="waveform",
            sample_id=sample.get("id"),
        )

        # --- Secondary waveform
        if "secondary_waveform" in sample:
            sample["secondary_waveform"], self._warned_secondary_8 = self._handle_waveform(
                sample["secondary_waveform"],
                self._warned_secondary_8,
                label="secondary_waveform",
                sample_id=sample.get("id"),
            )

        # --- Median beat
        if "median_beat" in sample:
            sample["median_beat"], self._warned_median_8 = self._handle_waveform(
                sample["median_beat"],
                self._warned_median_8,
                label="median_beat",
                sample_id=sample.get("id"),
            )

        # --- Secondary median beat
        if "secondary_median_beat" in sample:
            sample["secondary_median_beat"], self._warned_secondary_median_8 = self._handle_waveform(
                sample["secondary_median_beat"],
                self._warned_secondary_median_8,
                label="secondary_median_beat",
                sample_id=sample.get("id"),
            )

        return sample

class Resample(object):
    """
    Resamples ECG signals (and aligned time arrays) to a fixed sampling frequency.

    Supports:
        waveform: [leads, samples] or [segments, leads, samples]
        time:     [samples] or [segments, samples]
    """

    def __init__(self, sample_freq: int):
        self.sample_freq = int(sample_freq)

    def _resample_np(self, wf, samplebase):

        length = wf.shape[-1]
        duration = length / samplebase
        out_length = int(round(duration * self.sample_freq))

        x = np.arange(length) / samplebase
        xnew = np.arange(out_length) / self.sample_freq

        f = interpolate.interp1d(
            x,
            wf,
            axis=-1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        return f(xnew)

    def _resample_torch(self, wf, samplebase):
        length = wf.shape[-1]
        duration = length / samplebase
        out_length = int(round(duration * self.sample_freq))

        return torch.nn.functional.interpolate(
            wf.reshape(-1, 1, length),
            size=out_length,
            mode="linear",
            align_corners=False,
        ).reshape(*wf.shape[:-1], out_length)

    def _resample(self, wf, samplebase):
        if isinstance(wf, np.ndarray):
            return self._resample_np(wf, samplebase)
        if isinstance(wf, torch.Tensor):
            return self._resample_torch(wf, samplebase)

        raise TypeError("Unsupported waveform type")

    def _resample_time(self, t, samplebase):
        """
        Resample datetime64[us] array along last axis.
        """

        if not np.issubdtype(t.dtype, np.datetime64):
            raise TypeError("time must be datetime64")

        original_ndim = t.ndim

        if t.ndim == 1:
            t = t[None, :]  # [1, samples]

        if t.ndim not in (2, 3):
            raise ValueError("time must be 1D, 2D, or 3D")

        # reduce possible [segments, leads, samples] → [segments, samples]
        if t.ndim == 3:
            t = t[:, 0, :]

        length = t.shape[-1]
        duration = length / samplebase
        out_length = int(round(duration * self.sample_freq))

        # convert to float seconds relative to first sample
        t0 = t[..., 0:1]
        t_sec = (t - t0).astype("timedelta64[us]").astype(np.float64) / 1e6

        x = np.arange(length) / samplebase
        xnew = np.arange(out_length) / self.sample_freq

        f = interpolate.interp1d(
            x,
            t_sec,
            axis=-1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        t_sec_new = f(xnew)
        t_new = t0 + (t_sec_new * 1e6).astype("timedelta64[us]")

        if original_ndim == 1:
            t_new = t_new[0]

        return t_new


    def _process_waveform(self, wf, samplebase):
        original_ndim = wf.ndim

        if wf.ndim == 2:
            wf = wf[None, ...]

        if wf.ndim != 3:
            raise ValueError("Waveform must be 2D or 3D")

        expected_samples = int(10 * self.sample_freq)
        actual_duration = wf.shape[-1] / samplebase

        # if abs(actual_duration - 10.0) > 1e-3:
        #     raise AssertionError(
        #         f"Waveform is not 10 seconds (duration={actual_duration:.3f}s)"
        #     )

        if samplebase != self.sample_freq:
            wf = self._resample(wf, samplebase)

        if wf.shape[-1] != expected_samples:
            raise AssertionError(
                f"Expected {expected_samples} samples, got {wf.shape[-1]}"
            )

        if original_ndim == 2:
            wf = wf[0]

        return wf

    # ---------- call ----------

    def __call__(self, sample):
        samplebase = float(sample["samplebase"])

        sample["waveform"] = self._process_waveform(sample["waveform"], samplebase)

        if ("waveform" in sample and "secondary_waveform" in sample 
            and sample["waveform"].shape != sample["secondary_waveform"].shape):
            sample["secondary_waveform"] = self._process_waveform(
                sample["secondary_waveform"], samplebase
            )

        if "time" in sample:
            sample["time"] = self._resample_time(sample["time"], samplebase)

        sample["samplebase"] = self.sample_freq
        return sample


class ZeroPadder(object):
    """
    Apply zero padding to the ECG signal.

    write docstring
    """

    def __init__(self, target_length, pad_unit="samples", mode="middle"):
        assert pad_unit in [
            "samples",
            "seconds",
        ], "pad_unit must be 'samples' or 'seconds'."
        assert mode in [
            "start",
            "left",
            "end",
            "right",
            "middle",
            "center",
        ], "mode must be 'start'/'left', 'end'/'right', or 'middle'/'center'."
        self.target_length = target_length
        self.pad_unit = pad_unit
        self.mode = mode

    def _get_target_samples(self, samplebase):
        """Convert seconds to samples if needed."""
        if self.pad_unit == "seconds":
            return int(self.target_length * samplebase)
        return self.target_length

    def _zeropad(self, waveform, target_samples):
        """Apply zero padding based on mode."""
        current_length = waveform.shape[1]
        if current_length >= target_samples:
            return waveform
        pad_size = target_samples - current_length

        if isinstance(waveform, torch.Tensor):
            pad_func = torch.zeros
        else:
            pad_func = np.zeros

        if self.mode in ["end", "right"]:
            padded_waveform = pad_func((waveform.shape[0], pad_size))
            return (
                torch.cat([waveform, padded_waveform], dim=1)
                if isinstance(waveform, torch.Tensor)
                else np.concatenate([waveform, padded_waveform], axis=1)
            )

        elif self.mode in ["start", "left"]:
            padded_waveform = pad_func((waveform.shape[0], pad_size))
            return (
                torch.cat([padded_waveform, waveform], dim=1)
                if isinstance(waveform, torch.Tensor)
                else np.concatenate([padded_waveform, waveform], axis=1)
            )

        elif self.mode in ["middle", "center"]:
            left_pad = pad_size // 2
            right_pad = pad_size - left_pad
            left_pad_array = pad_func((waveform.shape[0], left_pad))
            right_pad_array = pad_func((waveform.shape[0], right_pad))
            return (
                torch.cat([left_pad_array, waveform, right_pad_array], dim=1)
                if isinstance(waveform, torch.Tensor)
                else np.concatenate([left_pad_array, waveform, right_pad_array], axis=1)
            )

    def __call__(self, sample):
        """Apply zero padding to a sample."""
        waveform = sample["waveform"]
        samplebase = int(sample["samplebase"])
        target_samples = self._get_target_samples(samplebase)
        sample["waveform"] = self._zeropad(waveform, target_samples)
        return sample


class ECGTrimmer(object):
    """
    Trim an ECG signal to a target length.

    Can trim in terms of:
    - 'samples' (fixed number of points)
    - 'seconds' (converts duration to number of samples using samplebase)

    Supports different trimming modes:
    - 'start' / 'left': Keep the end, remove from the start
    - 'end' / 'right': Keep the start, remove from the end
    - 'middle' / 'center': Remove equally from both ends
    """

    def __init__(self, target_length, trim_unit="samples", mode="middle"):
        assert trim_unit in [
            "samples",
            "seconds",
        ], "trim_unit must be 'samples' or 'seconds'."
        assert mode in [
            "start",
            "left",
            "end",
            "right",
            "middle",
            "center",
        ], "mode must be 'start'/'left', 'end'/'right', or 'middle'/'center'."

        self.target_length = target_length
        self.trim_unit = trim_unit
        self.mode = mode

    def _get_target_samples(self, samplebase):
        """Convert seconds to samples if needed."""
        if self.trim_unit == "seconds":
            return int(self.target_length * samplebase)
        return self.target_length

    def _trim_signal(self, waveform, target_samples):
        """Trim the signal based on the mode."""
        current_length = waveform.shape[1]
        if current_length <= target_samples:
            return waveform

        trim_size = current_length - target_samples

        if self.mode in ["end", "right"]:  # Trim from the end
            return waveform[:, :target_samples]

        elif self.mode in ["start", "left"]:  # Trim from the start
            return waveform[:, -target_samples:]

        elif self.mode in ["middle", "center"]:  # Trim equally from both sides
            left_trim = trim_size // 2
            right_trim = trim_size - left_trim
            return waveform[:, left_trim : current_length - right_trim]

    def __call__(self, sample):
        """Apply trimming to a sample."""
        waveform = sample["waveform"]
        samplebase = sample["samplebase"]
        target_samples = self._get_target_samples(samplebase)
        sample["waveform"] = self._trim_signal(waveform, target_samples)
        return sample


class ECGStandardizer(object):
    """
    Trim or pad ECG signals to a fixed length.
    Combines ECGTrimmer and ZeroPadder classes.
    """

    def __init__(self, target_length, unit="samples", mode="end"):
        self.target_length = target_length
        self.unit = unit
        self.mode = mode
        self.padder = ZeroPadder(target_length, unit, mode)
        self.trimmer = ECGTrimmer(target_length, unit, mode)

    def __call__(self, sample):
        waveform = sample["waveform"]
        samplebase = int(sample["samplebase"])

        target_samples = (
            self.target_length
            if self.unit == "samples"
            else int(self.target_length * samplebase)
        )

        if waveform.shape[1] < target_samples:
            return self.padder(sample)
        elif waveform.shape[1] > target_samples:
            return self.trimmer(sample)
        return sample  # No change needed


class NotchFilter(object):
    """
    Filter ECG by applying a Notch bandpass filter
    """

    def __init__(self, remove_freq, quality=30, fs=500):
        """Initializes the resample transformation."""

        self.b, self.a = signal.iirnotch(remove_freq, quality, fs)

    def __call__(self, sample):
        waveform = sample["waveform"]
        sample["waveform"] = np.ascontiguousarray(
            signal.filtfilt(self.b, self.a, waveform, axis=1, method="pad")
        )

        return sample


class AltNotchFilter(object):
    """
    Filter ECG by applying a Notch bandpass filter
    """

    def __init__(self, remove_freq, fs=500):
        """Initializes the resample transformation."""

        self.b = np.ones(int(fs / remove_freq))
        self.a = [len(self.b)]
        # self.b, self.a = signal.iirnotch(remove_freq, 30, fs)

    def __call__(self, sample):
        waveform = sample["waveform"]
        sample["waveform"] = np.ascontiguousarray(
            signal.filtfilt(self.b, self.a, waveform, axis=1, method="pad")
        )

        return sample


class Masker(object):

    def __init__(self, mask_func, use_seed, set_repeat=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.set_repeat = set_repeat

    def __call__(self, sample):
        shape = np.array(sample["waveform"].shape)
        seed = tuple(map(ord, sample["filename"])) if self.use_seed else None
        mask, acc = self.mask_func[0](shape, seed=seed)

        mask = mask.to(sample["waveform"].device)
        masked_x = sample["waveform"].clone()
        if self.set_repeat:
            for lead in range(masked_x.shape[0]):
                signal = masked_x[lead]  # (time,)
                mask_lead = mask[lead]  # (time,)

                unmasked_vals = signal[mask_lead == 1]
                num_to_fill = (mask_lead == 0).sum()

                if num_to_fill == 0:
                    continue  # nothing to fill for this lead

                if unmasked_vals.numel() == 0:
                    raise ValueError(
                        f"Lead {lead} has no unmasked values to repeat from."
                    )

                # Efficient strategy
                if unmasked_vals.numel() >= num_to_fill:
                    fill_vals = unmasked_vals[:num_to_fill]
                else:
                    repeats = (
                        num_to_fill + unmasked_vals.numel() - 1
                    ) // unmasked_vals.numel()
                    fill_vals = unmasked_vals.repeat(repeats)[:num_to_fill]

                signal[mask_lead == 0] = fill_vals
                masked_x[lead] = signal
        else:
            masked_x = (
                sample["waveform"] * mask + 0.0
            )  # the + 0.0 removes the sign of the zeros
        sample["masked_waveform"] = masked_x
        sample["layout"] = acc
        sample["mask"] = mask
        if "secondary_waveform" in sample:
            shape = np.array(sample["secondary_waveform"].shape)
            seed = (
                tuple(map(ord, sample["secondary_filename"])) if self.use_seed else None
            )
            mask, acc = self.mask_func[0](shape, seed=seed)

            mask = mask.to(sample["secondary_waveform"].device)
            masked_x = sample["secondary_waveform"].clone()
            if self.set_repeat:
                for lead in range(masked_x.shape[0]):
                    signal = masked_x[lead]  # (time,)
                    mask_lead = mask[lead]  # (time,)

                    unmasked_vals = signal[mask_lead == 1]
                    num_to_fill = (mask_lead == 0).sum()

                    if num_to_fill == 0:
                        continue  # nothing to fill for this lead

                    if unmasked_vals.numel() == 0:
                        raise ValueError(
                            f"Lead {lead} has no unmasked values to repeat from."
                        )

                    # Efficient strategy
                    if unmasked_vals.numel() >= num_to_fill:
                        fill_vals = unmasked_vals[:num_to_fill]
                    else:
                        repeats = (
                            num_to_fill + unmasked_vals.numel() - 1
                        ) // unmasked_vals.numel()
                        fill_vals = unmasked_vals.repeat(repeats)[:num_to_fill]

                    signal[mask_lead == 0] = fill_vals
                    masked_x[lead] = signal
            else:
                masked_x = (
                    sample["secondary_waveform"] * mask + 0.0
                )  # the + 0.0 removes the sign of the zeros
            sample["secondary_masked_waveform"] = masked_x
            sample["secondary_layout"] = acc
            sample["secondary_mask"] = mask

        return sample


class PolyFilter(object):
    """Filter ECG by subtracting a polynomial.

    Attributes:
        order (int): The order of the filter, usually 2, 3 or 4.
    """

    def __init__(self, order):
        """Initializes the resample transformation."""
        self.order = int(order)

    def __call__(self, sample):
        waveform = sample["waveform"]
        ecg_out = np.zeros_like(waveform)

        for lead in range(waveform.shape[0]):
            yp = Polynomial.fit(
                np.linspace(0, 10, waveform.shape[1]), waveform[lead, :], self.order
            )
            ecg_out[lead, :] = waveform[lead, :] - yp(
                np.linspace(0, 10, waveform.shape[1])
            )

        sample["waveform"] = ecg_out

        return sample


class ButterFilter(object):
    """
    Butterworth filter for ECG signals.

    Supports:
        2D: [leads, samples]
        3D: [segments, leads, samples]

    NaN-aware: each contiguous non-NaN run is filtered independently so
    that gaps (e.g. absent leads in layout ECGs) never introduce artificial
    discontinuities that cause ringing / edge artefacts.
    """

    def __init__(self, lowcut=None, highcut=None, order=3, fs=500):
        import scipy.signal as signal

        if lowcut is None and highcut is None:
            raise ValueError("At least one of lowcut or highcut must be set")

        if highcut is None:
            self.b, self.a = signal.butter(order, lowcut, btype="highpass", fs=fs)
        elif lowcut is None:
            self.b, self.a = signal.butter(order, highcut, btype="lowpass", fs=fs)
        else:
            self.b, self.a = signal.butter(
                order, [lowcut, highcut], btype="bandpass", fs=fs
            )

        # filtfilt needs at least padlen+1 samples; padlen = 3*max(len(a),len(b)-1)
        # For order-N butter: len(a)==len(b)==N+1, so padlen = 3*N
        self._min_seg_len = 3 * max(len(self.a), len(self.b) - 1) + 1

    def _contiguous_runs(self, mask):
        """
        Yield (start, stop) index pairs for every contiguous True run in *mask*.
        Uses numpy diff so there is no Python loop over samples.
        """
        import numpy as np

        if not mask.any():
            return

        # Pad with False on both ends so leading/trailing runs are caught
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        stops = np.where(diff == -1)[0]

        for s, e in zip(starts, stops):
            yield s, e          # half-open interval [s, e)

    def _filter_1d(self, x):
        """
        Filter a single 1-D lead, processing each contiguous non-NaN run
        independently.  Runs shorter than the minimum filtfilt length are
        left unfiltered rather than discarded so the original values survive.
        """
        import numpy as np
        import scipy.signal as signal

        out = x.copy()
        mask = ~np.isnan(x)

        for start, stop in self._contiguous_runs(mask):
            seg = x[start:stop]
            if len(seg) >= self._min_seg_len:
                out[start:stop] = signal.filtfilt(self.b, self.a, seg)
            # else: segment too short – leave values unchanged

        return out

    def _filter_2d(self, wf):
        import numpy as np

        out = np.empty_like(wf)
        for lead in range(wf.shape[0]):
            out[lead] = self._filter_1d(wf[lead])
        return np.ascontiguousarray(out)

    def _filter_3d(self, wf):
        import numpy as np

        out = np.empty_like(wf)
        for seg in range(wf.shape[0]):
            out[seg] = self._filter_2d(wf[seg])
        return out

    def __call__(self, sample):
        wf = sample["waveform"]

        if wf.ndim == 2:
            sample["waveform"] = self._filter_2d(wf)
        elif wf.ndim == 3:
            sample["waveform"] = self._filter_3d(wf)
        else:
            raise ValueError("Waveform must be 2D or 3D")

        return sample


class IgnorePeakAnnotations(object):
    """
    Ignore peak annotations removes the _PEAK post fix from segmentation annotations that are formatted like the LUDB
    dataset.
    """

    def __call__(self, sample: dict):
        if "annotation" not in sample:
            raise Exception(
                "'annotation' is not present in sample dictonary, please ensure it is present in all samples"
            )
        # Find indices of annotations containing '_PEAK'
        peaks = np.nonzero(
            np.core.defchararray.find(sample["annotation"].astype(str), "_PEAK") != -1
        )
        # Replace with first character of annotation (cast to unicode string of len 1)
        sample["annotation"][peaks] = sample["annotation"][peaks].astype("<U1")
        return sample


class AnnotationsToClass(object):
    """
    Converts annotations formatted as strings to their specified class index. The appropriate
    index is determined through the mapping dict passed to the function.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, sample):
        if "annotation" not in sample:
            raise Exception(
                "'annotation' is not present in sample dictonary, please ensure it is present in all samples"
            )
        class_annot = np.zeros_like(sample["annotation"], dtype=float)
        for k, v in self.mapping.items():
            class_annot[sample["annotation"].astype(str) == k] = v
        sample["annotation"] = class_annot

        return sample


class CombineLeadAnnotations(object):
    """
    Combines the annotations of multiple leads when they are the same. Labels as noise when all leads are noise.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, sample):
        class_annot = sample["annotation"].astype(np.float32)

        # If pred is noise set to nan (so that it is excluded in nan median)
        class_annot[class_annot == self.mapping["noise"]] = np.nan

        # Take median ignoring noisy leads
        class_annot_combined = np.nanmedian(class_annot, axis=0)

        # Correct for fact that argmax of all nan values is 0
        class_annot_combined[np.all(np.isnan(class_annot), axis=0)] = self.mapping[
            "noise"
        ]

        sample["annotation"] = class_annot_combined.astype(np.int64)

        return sample


class RandomChannel(object):
    """
    Returns a random channel from the ecg
    """

    def __init__(self, use_annotation: bool = False):
        self.use_annotation = use_annotation

    def __call__(self, sample):
        num_channels = sample["waveform"].shape[0]
        selected_channel = np.random.randint(0, num_channels)

        sample["waveform"] = np.expand_dims(sample["waveform"][selected_channel], 0)
        if self.use_annotation:
            sample["annotation"] = sample["annotation"][selected_channel]
        return sample


class BaselineCorrection(object):
    """
    Baseline correction for ECG-like signals.

    Supports:
        [leads, samples]
        [segments, leads, samples]
    """

    def __call__(self, sample):
        sample["waveform"] = self._correct(sample["waveform"])
        if "secondary_waveform" in sample:
            sample["secondary_waveform"] = self._correct(sample["secondary_waveform"])
        return sample

    @staticmethod
    def _correct(wf):
        import numpy as np

        if wf.ndim == 1:
            raise ValueError("Waveform is 1D (flattened). Expected 2D or 3D.")

        original_ndim = wf.ndim

        if wf.ndim == 2:
            wf = wf[None, ...]  # [1, leads, samples]

        if wf.ndim != 3:
            raise ValueError(f"Invalid waveform shape: {wf.shape}")

        wf = wf.astype(np.float32, copy=True)

        # FIX 1: exclude both NaN and zero-padded regions from the mask.
        # Previously `wf != 0` returned True for NaN (nan != 0 is True),
        # so NaN samples were silently included in the baseline calculation.
        mask = np.isfinite(wf) & (wf != 0)

        counts = mask.sum(axis=-1, keepdims=True)
        counts = np.maximum(counts, 1)

        # FIX 2: use np.where to zero out invalid entries before summing.
        # Previously `wf * mask` was used, but nan * False = nan in NumPy,
        # which poisoned the entire sum for any lead containing NaN values.
        baseline = np.where(mask, wf, 0.0).sum(axis=-1, keepdims=True) / counts

        wf -= baseline
        wf[~mask] = np.nan

        if original_ndim == 2:
            wf = wf[0]

        return wf


class SelectChannel(object):
    def __init__(self, use_annotation: bool = False):
        self.use_annotation = use_annotation

    def __call__(self, sample):
        selected_channel = sample["lead"]

        sample["waveform"] = np.expand_dims(sample["waveform"][selected_channel], 0)
        if self.use_annotation:
            sample["annotation"] = sample["annotation"][selected_channel]
        return sample


class BaselineDriftAugmentation:
    """
    Smooth polynomial baseline drift (Torch-only).
    """

    def __init__(
        self,
        height_std: float = 0.4,
        augment_prob: float = 0.95,
        num_points: int = 6,
        max_shift: int = 100,
        deg: int = 5,
        waveform_key: str = "waveform",
        use_seed: bool = True,
    ):
        self.height_std = height_std
        self.augment_prob = augment_prob
        self.num_points = num_points
        self.max_shift = max_shift
        self.deg = deg
        self.waveform_key = waveform_key
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        waveform = sample[self.waveform_key]
        device, dtype = waveform.device, waveform.dtype
        _, n_samples = waveform.shape

        seed = make_seed(sample, self.waveform_key, "baseline_drift", self.use_seed)

        with temp_seed(self.rng, seed):
            if self.rng.random() > self.augment_prob:
                return sample

            interval = n_samples // self.num_points
            points = torch.arange(self.num_points + 1, device=device) * interval
            points[-1] = n_samples - 1

            shifts = torch.tensor(
                self.rng.normal(scale=self.max_shift, size=self.num_points - 1),
                device=device,
                dtype=torch.long,
            )

            points[1:-1] += shifts
            points.clamp_(0, n_samples - 1)

            heights = torch.tensor(
                self.rng.normal(scale=self.height_std, size=self.num_points + 1),
                device=device,
                dtype=dtype,
            )

        x = points.float()
        y = heights
        vander = torch.vander(x, self.deg + 1, increasing=True)
        coeffs = torch.linalg.lstsq(vander, y).solution

        t = torch.arange(n_samples, device=device, dtype=dtype)
        poly = torch.sum(
            coeffs * t[:, None] ** torch.arange(self.deg + 1, device=device),
            dim=1,
        )

        waveform = waveform + poly[None, :]

        if self.waveform_key == "masked_waveform":
            waveform = waveform * sample["mask"]

        sample[self.waveform_key] = waveform
        return sample


class NoiseAugmentation:
    """
    Sinusoidal noise (Torch-only).
    """

    def __init__(
        self,
        measured_seconds: int = 10,
        max_height_std: float = 0.1,
        augment_prob: float = 0.2,
        freq_min: int = 25,
        freq_max: int = 100,
        waveform_key: str = "waveform",
        use_seed: bool = True,
    ):
        self.measured_seconds = measured_seconds
        self.max_height_std = max_height_std
        self.augment_prob = augment_prob
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.waveform_key = waveform_key
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        waveform = sample[self.waveform_key]
        device, dtype = waveform.device, waveform.dtype
        n_channels, n_samples = waveform.shape

        seed = make_seed(sample, self.waveform_key, "sine_noise", self.use_seed)

        with temp_seed(self.rng, seed):
            apply_mask = torch.tensor(
                self.rng.random(n_channels) < self.augment_prob,
                device=device,
            )

            if not apply_mask.any():
                return sample

            amplitude = torch.tensor(
                self.rng.random(n_channels) * self.max_height_std,
                device=device,
                dtype=dtype,
            )

            frequency = torch.tensor(
                self.rng.randint(self.freq_min, self.freq_max, size=n_channels),
                device=device,
                dtype=dtype,
            )

        t = torch.linspace(
            0,
            self.measured_seconds,
            n_samples,
            device=device,
            dtype=dtype,
        )

        noise = amplitude[:, None] * torch.sin(2 * torch.pi * frequency[:, None] * t)

        waveform[apply_mask] += noise[apply_mask]

        if self.waveform_key == "masked_waveform":
            waveform = waveform * sample["mask"]

        sample[self.waveform_key] = waveform
        return sample


class SpatialCut(object):
    def __init__(self, old_len: int, new_len: int, mode: str = "equal"):
        assert (
            new_len < old_len
        ), "Old length must be bigger then new lenght, use Resample() if you want the increase the sample length"
        self.mode = mode

        self.diff = old_len - new_len
        self.old_len = old_len
        self.new_len = new_len

        if self.mode == "equal":
            self.onset = self.diff // 2
            self.offset = self.diff - self.onset

    def __call__(self, sample):
        if self.mode == "equal":
            sample["waveform"] = sample["waveform"][:, self.onset : -self.offset]

        elif self.mode == "random":
            onset = int(np.random.randint(low=0, high=self.old_len - self.diff))
            sample["waveform"] = sample["waveform"][:, onset : onset + self.new_len]

        elif self.mode == "random_multi":
            # TODO, just 8 cuts for now
            step = self.new_len // 8
            new_waveform = np.zeros((sample["waveform"].shape[0], self.new_len))
            for i in range(8):
                onset = int(np.random.randint(low=0, high=self.old_len - step))
                new_waveform[:, i * step : i * step + step] = sample["waveform"][
                    :, onset : onset + step
                ]
            sample["waveform"] = new_waveform
        return sample


class PermuteChannelAugmentation(object):
    def __init__(self, augment_prob: float = 0.1):
        self.augment_prob = augment_prob

    def __call__(self, sample):
        if np.random.rand(1) > self.augment_prob:
            return sample

        sample["waveform"] = np.random.permutation(sample["waveform"])
        return sample


class HorizontalFlipAugmentation(object):
    def __init__(self, augment_prob: float = 0.01):
        self.augment_prob = augment_prob

    def __call__(self, sample):
        augment_lead_mask = (
            np.random.rand(sample["waveform"].shape[0]) < self.augment_prob
        )

        if augment_lead_mask.sum() == 0:
            return sample

        sample["waveform"][augment_lead_mask] *= -1

        return sample


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    """Temporarily set the seed of a numpy random number generator.

    Parameters
    ----------
    rng : np.random.Generator
        The numpy random number generator to modify.
    seed : Optional[Union[int, Tuple[int, ...]]], optional
        The seed to set, by default None.
    """
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


def make_seed(sample, waveform_key: str, transform_key: str, use_seed: bool):
    if not use_seed:
        return None
    return tuple(map(ord, sample["filename"] + waveform_key + transform_key))


class GaussianNoiseAugmentation:
    """
    Gaussian noise (Torch-only).
    """

    def __init__(
        self,
        max_std: float = 0.05,
        augment_prob: float = 0.3,
        waveform_key: str = "waveform",
        use_seed: bool = True,
    ):
        self.max_std = max_std
        self.augment_prob = augment_prob
        self.waveform_key = waveform_key
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        waveform = sample[self.waveform_key]
        device, dtype = waveform.device, waveform.dtype
        n_channels, n_samples = waveform.shape

        seed = make_seed(sample, self.waveform_key, "gaussian_noise", self.use_seed)

        with temp_seed(self.rng, seed):
            apply_mask = torch.tensor(
                self.rng.random(n_channels) < self.augment_prob,
                device=device,
            )

            if not apply_mask.any():
                return sample

            std = torch.tensor(
                self.rng.random(n_channels) * self.max_std,
                device=device,
                dtype=dtype,
            )

        noise = (
            torch.randn(
                n_channels,
                n_samples,
                device=device,
                dtype=dtype,
            )
            * std[:, None]
        )

        waveform[apply_mask] += noise[apply_mask]

        if self.waveform_key == "masked_waveform":
            waveform = waveform * sample["mask"]

        sample[self.waveform_key] = waveform
        return sample


class ZeroLeadAugmentation(object):
    def __init__(
        self,
        height_std: float = 0.1,
        augment_prob: float = 0.05,
        multi_channel: bool = True,
    ):
        self.augment_prob = augment_prob
        self.height_std = height_std

    def __call__(self, sample):
        augment_lead_mask = (
            np.random.rand(sample["waveform"].shape[0]) < self.augment_prob
        )

        if augment_lead_mask.sum() == 0:
            return sample

        sample["waveform"][augment_lead_mask, :] = 0
        return sample


class RandomShiftAugmentation(object):
    def __init__(self, shift_size: int, mirror_padding: bool = False):
        self.shift_size = shift_size
        self.mirror_padding = mirror_padding

    def __call__(self, sample):
        shift = int(np.random.normal(size=1) * self.shift_size)

        if shift == 0:
            return sample

        if self.mirror_padding:
            sample["waveform"] = (
                np.roll(sample["waveform"], shift, axis=-1)
                if isinstance(sample["waveform"], np.ndarray)
                else torch.roll(sample["waveform"], shift, dims=-1)
            )
        else:
            if shift > 0:
                sample["waveform"][:, shift:] = sample["waveform"][:, :-shift]
                sample["waveform"][:, :shift] = 0
            else:
                sample["waveform"][:, :shift] = sample["waveform"][:, -shift:]
                sample["waveform"][:, shift:] = 0
        return sample


class Normalize(object):
    def __init__(self, means, stds, min_std=0.1):
        self.means = means
        self.stds = stds
        self.stds[self.stds < min_std] = min_std

    def __call__(self, sample):
        sample["waveform"] = (sample["waveform"] - self.means) / self.stds
        return sample


class CenterPerLead(object):
    def __call__(self, sample):
        self.means = np.mean(sample["waveform"], axis=1, keepdims=True)
        sample["waveform"] = sample["waveform"] - self.means

        return sample


class CenterPerLead(object):
    def __call__(self, sample):
        self.means = np.mean(sample["waveform"], axis=1, keepdims=True)
        sample["waveform"] = sample["waveform"] - self.means

        return sample


class RandomButterFilter(object):
    """
    Filter ECG by applying a Butter bandpass using random settings

    Attributes:
        lowcut (int): Filter frequencies below this value
        highcut (int): Filter frequencies above this value
        order (int): The order of the filter, usually 2, 3 or 4.
        fs (int): Sampling frequency
    """

    def __init__(
        self,
        augment_prob: float = 0.5,
        lowcut_min=0.1,
        lowcut_max=1,
        highcut_min=50,
        highcut_max=150,
        order_min=2,
        order_max=4,
        fs=500,
    ):
        """Initializes the resample transformation."""
        self.augment_prob = augment_prob

        self.lowcut_min = lowcut_min
        self.lowcut_max = lowcut_max
        self.highcut_min = highcut_min
        self.highcut_max = highcut_max
        self.order_min = order_min
        self.order_max = order_max
        self.fs = fs

    def __call__(self, sample):
        if np.random.rand(1) > self.augment_prob:
            return sample

        lowcut = self.lowcut_min + int(
            np.random.rand(1) * (self.lowcut_max - self.lowcut_min)
        )
        highcut = self.highcut_min + int(
            np.random.rand(1) * (self.highcut_max - self.highcut_min)
        )
        order = self.order_min + int(
            np.random.rand(1) * (self.order_max - self.order_min)
        )

        b, a = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=self.fs)

        waveform = sample["waveform"]
        sample["waveform"] = np.ascontiguousarray(
            signal.filtfilt(b, a, waveform, axis=1)
        )

        return sample


class MuLawQuantization(object):
    def __init__(
        self,
        mu,
        encode: bool = True,
        onehot: bool = True,
        normalization_cutoff: float = 5.0,
    ):
        self.mu = mu
        self.normalization_cutoff = normalization_cutoff
        self.onehot = onehot
        self.encode = encode

    def __call__(self, sample):
        x = sample["waveform"]

        x = x.clamp(min=-self.normalization_cutoff, max=self.normalization_cutoff)
        x /= self.normalization_cutoff

        if self.encode:
            sign_x = (x + 1e-10).sign()
            mu_law_res = sign_x * ((1 + self.mu * x.abs()).log() / np.log(1 + self.mu))
            quantized = torch.bucketize(mu_law_res, torch.linspace(-1, 1, self.mu))
            sample["waveform"] = (
                torch.nn.functional.one_hot(quantized, num_classes=self.mu).type(
                    x.type()
                )
                if self.onehot
                else quantized
            )
            return sample

        sign_x = (x + 1e-10).sign()
        out = torch.zeros_like(x) + self.mu + 1

        return sign_x * (out ** x.abs() - 1) / self.mu


class Reshape(object):
    """Reshape input tensor to given size"""

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        sample["waveform"] = sample["waveform"].reshape(self.shape)
        return sample


class PermutationAugmentation(object):
    """
    Divide the ECG into segments and permute them to create
    an augmented sample (as seen in arxiv.org/abs/2206.07656)
    """

    def __init__(self, segments: int = 10, augment_prob: float = 0.5):
        self.segments = segments
        self.augment_prob = augment_prob

    def __call__(self, sample: np.ndarray):
        if np.random.rand(1) > self.augment_prob:
            return sample

        seg_len = sample["waveform"].shape[1] // self.segments
        seg_len_last = seg_len + (sample["waveform"].shape[1] % self.segments)

        seg_perm = np.random.permutation(np.arange(self.segments))
        idx_perm = []
        for seg in seg_perm:
            if seg == self.segments - 1:
                idx_perm.append(
                    np.arange(seg * seg_len, seg * seg_len + seg_len_last, 1)
                )
            else:
                idx_perm.append(np.arange(seg * seg_len, (seg + 1) * seg_len, 1))
        idx_perm = np.hstack(idx_perm)

        sample["waveform"] = sample["waveform"][:, idx_perm]

        return sample


class Transpose(object):
    """
    Transpose the ECG
    """

    def __call__(self, sample):
        sample["waveform"] = sample["waveform"].T


import numpy as np
from typing import Optional


class ECGNormalizer(object):
    """Normalizes ECG signals.
    Parameters
    ----------
    normalization_type : str, optional
        One of:
        - "zscore": (x - mean) / std
        - "minmax": (x - min) / (max - min)
        - "max": x / max
        - "mean": x / mean
        - "robust": (x - median) / MAD
        - None: no normalization
    Returns
    -------
    normalized_data : np.ndarray
    attrs : dict
        Statistics used in normalization.
    """

    def __init__(self, normalization_type: Optional[str] = None, global_stats: bool = False):
        self.normalization_type = normalization_type
        self.global_stats = global_stats  # True = match ECGFounder (scalar mean/std over all leads)
        self.eps = 1e-8

    def __extract_stats__(
        self, data: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, dict]:
        # shape: (n_leads, n_samples)
        if self.global_stats:
            # Match ECGFounder: scalar mean/std over all leads and samples combined
            mean_ = np.full(data.shape[0], data.mean())
            std_  = np.full(data.shape[0], data.std())
            min_  = np.full(data.shape[0], data.min())
            max_  = np.full(data.shape[0], data.max())
            median_ = np.full(data.shape[0], np.median(data))
        elif mask is not None:
            observed_counts = mask.sum(axis=1).clip(min=1.0)
            mean_ = (data * mask).sum(axis=1) / observed_counts
            std_ = np.sqrt(
                ((data - mean_[:, None]) ** 2 * mask).sum(axis=1) / observed_counts
            )
            min_ = np.where(mask.astype(bool), data, np.inf).min(axis=1)
            max_ = np.where(mask.astype(bool), data, -np.inf).max(axis=1)
            median_ = np.median(
                np.where(mask.astype(bool), data, mean_[:, None]), axis=1
            )
        else:
            min_ = data.min(axis=1)
            max_ = data.max(axis=1)
            mean_ = data.mean(axis=1)
            std_ = data.std(axis=1)
            median_ = np.median(data, axis=1)

        attrs = {
            "min": min_,
            "max": max_,
            "mean": mean_,
            "std": std_,
            "median": median_,
        }

        if self.normalization_type == "zscore":
            data = (data - mean_[:, None]) / (std_[:, None] + self.eps)
        elif self.normalization_type == "minmax":
            data = (data - min_[:, None]) / (max_[:, None] - min_[:, None] + self.eps)
        elif self.normalization_type == "mean":
            data = data / (mean_[:, None] + self.eps)
        elif self.normalization_type == "max":
            data = data / (max_[:, None] + self.eps)
        elif self.normalization_type == "robust":
            mad = np.median(np.abs(data - median_[:, None]), axis=1)
            data = (data - median_[:, None]) / (mad[:, None] + self.eps)
        elif self.normalization_type is None:
            pass
        else:
            raise ValueError(
                f"Unsupported normalization type: {self.normalization_type}"
            )

        return data, attrs

    def __call__(self, sample: dict) -> dict:
        if "waveform" in sample:
            target, t_attrs = self.__extract_stats__(sample["waveform"])
            sample["waveform"] = target
        else:
            t_attrs = None

        if "masked_waveform" in sample and "mask" in sample:
            prediction, p_attrs = self.__extract_stats__(
                sample["masked_waveform"], sample["mask"]
            )
            sample["masked_waveform"] = prediction
        else:
            p_attrs = None

        if "attrs" in sample:
            for prefix, attrs in [("target", t_attrs), ("prediction", p_attrs)]:
                if attrs is not None:
                    for key, val in attrs.items():
                        sample["attrs"][f"{prefix}_{key}"] = val

        for beat_key in ("median_beat", "secondary_median_beat"):
            if beat_key not in sample or t_attrs is None:
                continue

            beat = sample[beat_key]  # shape: (n_leads, n_beat_samples)

            if self.normalization_type == "zscore":
                beat = (beat - t_attrs["mean"][:, None]) / (t_attrs["std"][:, None] + self.eps)
            elif self.normalization_type == "minmax":
                beat = (beat - t_attrs["min"][:, None]) / (t_attrs["max"][:, None] - t_attrs["min"][:, None] + self.eps)
            elif self.normalization_type == "mean":
                beat = beat / (t_attrs["mean"][:, None] + self.eps)
            elif self.normalization_type == "max":
                beat = beat / (t_attrs["max"][:, None] + self.eps)
            elif self.normalization_type == "robust":
                median_ = t_attrs["median"]
                mad = np.median(np.abs(beat - median_[:, None]), axis=1)
                beat = (beat - median_[:, None]) / (mad[:, None] + self.eps)

            sample[beat_key] = np.nan_to_num(beat, nan=0.0)

        return sample
