"""
ECG datasets to use with ECGx.AI package.

MB Vessies and RR van de Leur
"""

import os
import torch
from torch.utils.data import Dataset
from scipy import io
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np
from typing import List, Union, Callable, Optional, Dict, Any, Tuple
import pandas as pd
from torchvision.transforms import Compose
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import subprocess
import wfdb
import struct
import warnings
from collections import defaultdict
import h5py
import random
import cv2
import xarray as xr
import json
from datetime import datetime, timedelta
from pathlib import Path
import pyedflib
import traceback

class DatasetBase(Dataset):
    """
    Only used to inherit some functions later on.
    """

    def get_dataframe(self):
        """
        Returns the dataframe that is being used in this dataset.
        """
        return self.dataset

    def query_dataset(
        self,
        query: List[tuple],
        max_num_samples: int = None,
        return_indices: bool = False,
    ):
        """
        Query dataset for specific class (combinations), returns samples (dict with wavefrom etc) or indices in dataset.

        Args:
            query: Defines what classes returned samples should (not) have e.g. [('rhythm_st', True), ('conduction_lbtb': False)]
            max_num_samples: Maximum number of samples to return from query (default: {None})
            return_indices: Wether to return samples (dict with waveform etc) or indices in dataset (default: {False})

        Returns:
            List of samples or list of indices that meet the query.
        """
        query = query if type(query) == list else [query]

        qs = " and ".join(
            [
                (
                    f"{param} == {val}"
                    if not isinstance(val, str)
                    else f'{param} == "{val}"'
                )
                for (param, val) in query
            ]
        )
        qres = self.dataset.query(qs)
        if max_num_samples and max_num_samples < qres.shape[0]:
            qres = qres[:max_num_samples]

        if return_indices:
            return [self.dataset.index.get_loc(qidx) for qidx in qres.index]

        return [
            self.__getitem__(self.dataset.index.get_loc(qidx)) for qidx in qres.index
        ]

    def print_stats(self):
        """Prints statistics of the dataset."""
        stats, full_size = self.get_stats()
        print("-- Dataset distribution -- ")
        print(f"Full size: {full_size}")
        for s in stats:
            print(f'["{s["class"]}"] - Num entries: {s["size"]} ({s["fraction"]:.3}%)')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)


class UniversalECGDataset(DatasetBase):
    """
    Universal ECG dataset in PyTorch Dataset format.

    This class defines a dataset that can be used to train PyTorch models
    using a wide range of different ECG formats. On initialization, this
    dataset format can be defined, or a custom function to load the data
    can be provided. During training, a dictionary is returned on every
    iteration with the following contents:

    {
        'waveform': numpy array with the ECG waveform
        'samplebase': int of the sampling frequency,
        'gain': float of the gain of the ECG, to convert it into millivolts,
        'id': the id of the sample,
    }

    Optionally, some other variables are also provided, when they exist
    in the dataframe. These can be used for transformations or loss functions
    later on.

    Attributes:
        dataset_function: Selects the function to use to load dataset samples. Predefined
            options are 'umcu', 'universal', 'wfdb', 'physionet' and 'physionet_numpy'. In the
            UMCU datasets, ECGs are saved in the numpy format using the following folder structure:
            `waveform_dir/ps/eu/doid/testid.npy`. In the universal datasets, files are also saved using
            the numpy format, but then all in one folder: `waveform_dir/filename.npy`. For the WFDB data,
            data is saved in the WFDF format. For the physionet
            datasets, files are saved either using the matlab format of the numpy format. It is also
            possible to provide your own function to load custom datasets. Inside this function you
            will have access to all columns in the provided dataset to correctly load your
            custom ECG file. Make sure you return the ECG with the channels or leads as the first dimension
            and the samples as the second dimension.
        waveform_dir: Path of the folder with the raw waveform files are stored.
        dataset: Pandas DataFrame with the dataset your are using. For all datasets, the columns
            SampleBase (the sampling frequency) and the Gain (to convert the signal to mV) are required.
            The UMCU dataset requires the PseudoID and TestID columns, while all other datasets
            only require a Filename column.
        transform: List of PyTorch transformations.
        labels: Name(s) of the y variable in the dataset,
            if a list of names is supplied a vector is returned.
        secondary_waveform_dir: Directory of a possible secondary waveform, can be used if multiple waveforms
            from different locations are required for a model to train. E.g. we can use it when we try
            to convert rhythm ECGs to median beat ECGs. To use the secondary waveform, the dataset should
            be saved in the same format as the main dataset.
        additional_dataset_function: (list of) function(s) that can be used to load additional data for the sample.
            The provided function(s) will be given the UniversalECGDataset object and the df row as input and
            should return a dictionary that will be merged with the sample dictonary returned by the default
            dataset. These functions will be called AFTER the initial loading of the waveform/labels and BEFORE
            the application of the transforms.
    """

    def __init__(
        self,
        dataset_function: Union[str, Callable],
        waveform_dir: Union[str, List[str]],
        dataset: pd.DataFrame,
        transform: Optional[Compose] = None,
        labels: Optional[Union[str, List[str]]] = None,
        secondary_waveform_dir: str = "",
        median_beat_dir: Optional[str] = None,
        secondary_median_beat_dir: Optional[str] = None,
        median_load_params: Optional[dict] = {
            "meta":True,
            "tables": True,
            "waveforms": True,
            "annotations":True,
        },
        segment_seconds: Optional[int] = 10,
        additional_dataset_function: Optional[Union[Callable, List[Callable]]] = None,
        log_figures: Optional[int] = None,
    ):
        """
        Initializes the ECG datasets.
        """
        assert dataset_function in [
            "umcu",
            "umcu_xECG",
            "umcu_dig",
            "umcu_sl",
            "universal",
            "universal_holter",
            "physionet",
            "physionet_numpy",
            "wfdb",
            "hdf5",
        ] or isinstance(dataset_function, Callable)
        assert (
            isinstance(transform, Compose) or not transform
        ), "Transform should be a torchvision Compose object"

        if dataset_function == "umcu":
            if not ("PseudoID" in dataset and "TestID" in dataset):
                raise ValueError(
                    "Please provide a dataframe with both PseudoID and TestID columns when using the UMCU format."
                )
        elif dataset_function == "umcu_xECG":
            if not (
                "PseudoID" in dataset and "TestID" in dataset and "PhaseID" in dataset
            ):
                raise ValueError(
                    "Please provide a dataframe with both PseudoID and TestID columns when using the UMCU format."
                )
        elif dataset_function in [
            "universal",
            "physionet",
            "physionet_numpy",
            "universal_holter",
        ]:
            if "Filename" not in dataset:
                raise ValueError(
                    "Please provide a dataframe with Filename column when using this format."
                )

        self.dataset = dataset
        self.waveform_dir = waveform_dir
        self.secondary_waveform_dir = secondary_waveform_dir
        self.transform = transform
        self.labels = labels
        self.label_set = self.dataset[labels] if labels else None
        self.dataset_function = dataset_function
        self.additional_dataset_function = additional_dataset_function
        self.median_beat_dir = median_beat_dir
        self.secondary_median_beat_dir = secondary_median_beat_dir
        self.median_load_params = median_load_params
        self.segment_seconds = segment_seconds
        self.log_figures = log_figures
        if isinstance(self.additional_dataset_function, Callable):
            self.additional_dataset_function = [self.additional_dataset_function]
        if self.log_figures:
            self.log_images = random.sample(range(len(self.dataset)), self.log_figures)
        else:
            self.log_images = []
        self.index_map: list[tuple[int, int]] = []
        self.file_time_info = {}
        for file_idx, row in self.dataset.iterrows():
            if self.dataset_function == "universal_holter":
                filename = row["Filename"]
                path = os.path.join(self.waveform_dir, f"{filename}.EDF")

                n_segments, start_time, end_time = self.read_holter_metadata(
                    path,
                    segment_seconds=self.segment_seconds,
                )
                self.file_time_info[file_idx] = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "n_segments": n_segments,
                }

                for seg_idx in range(n_segments):
                    self.index_map.append((file_idx, seg_idx))
            elif self.dataset_function == "umcu_sl":
                duration = row["Duration"]
                n_segments = duration // 10
                for seg_idx in range(n_segments):
                    self.index_map.append((file_idx, seg_idx))
            else:
                self.index_map.append((file_idx, None))

    def train_test_split(
        self,
        ratio: float = 0.1,
        shuffle: bool = True,
        group_by: Optional[str] = "PseudoID",
        random_state: int = 1234,
    ):
        """
        Splits the current dataset into a (grouped) train and testset.

        Args:
            ratio: The fraction of the dataset that should be considered the testset (default: 0.1)
            shuffle: Wether the shuffle the dataset before splitting (default: True) -
                Note that the dataset will always be shuffled if the group_by parameter is set
            group_by: Variable to which the dataset should be grouped before splitting. e.g. setting
                this paramater to 'PseudoID' (default) will ensure no patients are present in both the
                train and testset. For the grouping to work the group_by parameter has to be present
                in the original dataframe used to create the dataset.
            random_state: Random seed used for shuffling

        Returns:
            A trainset and a testset, both as UniversalECGDataset objects
        """

        if group_by:
            splitter = GroupShuffleSplit(1, test_size=ratio, random_state=random_state)
            trainset_indices, testset_indices = next(
                splitter.split(self.dataset, groups=self.dataset[group_by])
            )
            trainset_df, testset_df = (
                self.dataset.iloc[trainset_indices],
                self.dataset.iloc[testset_indices],
            )
        else:
            trainset_df, testset_df = train_test_split(
                self.dataset,
                test_size=ratio,
                shuffle=shuffle,
                random_state=random_state,
            )

        trainset = UniversalECGDataset(
            self.dataset_function,
            self.waveform_dir,
            trainset_df.reset_index(drop=True),
            self.transform,
            self.labels,
            self.secondary_waveform_dir,
            self.median_beat_dir,
            self.secondary_median_beat_dir,
            self.median_load_params,
            self.segment_seconds,
            self.additional_dataset_function,
            self.log_figures
        )

        testset = UniversalECGDataset(
            self.dataset_function,
            self.waveform_dir,
            testset_df.reset_index(drop=True),
            self.transform,
            self.labels,
            self.secondary_waveform_dir,
            self.median_beat_dir,
            self.secondary_median_beat_dir,
            self.median_load_params,
            self.segment_seconds,
            self.additional_dataset_function,
            self.log_figures
        )

        return trainset, testset

    @staticmethod
    def _load_median_core(
        zarr_path: str,
        load: dict,
        segment_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Core loading logic for median Zarr stores.
        Based on loadUMCUMedian: tries consolidated=True first, falls back to False.
        """
        out: Dict[str, Any] = {}
 
        group_prefix = "" if segment_idx is None else f"segments/segment_{segment_idx:03d}/"
 
        def normalize(req, available):
            if req is False:
                return []
            if req is True:
                return list(available)
            return list(req)
 
        def safe_open(group: str):
            try:
                return xr.open_zarr(zarr_path, group=group, consolidated=True)
            except Exception:
                return xr.open_zarr(zarr_path, group=group, consolidated=False)
 
        # ================= META =================
        if load.get("meta", False):
            ds = safe_open("meta")
            out["meta"] = dict(ds.attrs)
 
        # ================= TABLES =================
        table_req = load.get("tables", False)
        if table_req:
            ds = safe_open(f"{group_prefix}tables")
 
            available = [
                f"{prefix}_df_{col}"
                for v in ds.data_vars
                for prefix, col in [v.split("__", 1)]
            ]
            wanted = normalize(table_req, available)
 
            complex_out: Dict[str, Any] = {}
            noise_out: Dict[str, Any] = {}
 
            for v in ds.data_vars:
                prefix, col = v.split("__", 1)
                req_key = f"{prefix}_df_{col}"
 
                if req_key not in wanted:
                    continue
 
                da = ds[v]
                t = da.attrs.get("type")
 
                if t == "categorical":
                    val = {
                        "codes": da.values.astype(np.int16, copy=False),
                        "map": np.asarray(da.attrs["categories"], dtype=str),
                        "missing": da.attrs.get("missing"),
                    }
                elif t == "boolean":
                    val = da.values.astype(np.bool_, copy=False)
                elif t == "integer":
                    # stored as int16 — load as int16 rather than int64 (8× smaller)
                    val = da.values.astype(np.int16, copy=False)
                elif t == "float":
                    # stored as float16 — load as float16 rather than float32 (2× smaller)
                    val = da.values.astype(np.float16, copy=False)
                else:
                    continue
 
                if prefix == "complex":
                    complex_out[col] = val
                else:
                    noise_out[col] = val
 
            if complex_out:
                out["complex_df"] = complex_out
            if noise_out:
                out["noise_df"] = noise_out
 
        # ================= WAVEFORMS =================
        wave_req = load.get("waveforms", False)
        if wave_req:
            ds = safe_open(f"{group_prefix}waveforms")
            wanted = normalize(wave_req, ds.data_vars)
 
            for name in wanted:
                if name in ds:
                    # stored as float16 — return as-is to avoid doubling memory on load
                    out[name] = ds[name].values
 
        # ================= ANNOTATIONS =================
        ann_req = load.get("annotations", False)
        if ann_req:
            ds = safe_open(f"{group_prefix}annotations")
            wanted = normalize(ann_req, ds.data_vars.keys())
 
            if "fiducials" in wanted and "fiducials" in ds.data_vars:
                da = ds["fiducials"]
                fid = da.values.astype(np.float32, copy=False)
                out["fiducials"] = fid
                out["fiducials_ndim"] = fid.ndim
                out["fiducials_dims"] = tuple(da.dims)
 
            ci = {}
            for name in wanted:
                if name == "fiducials":
                    continue
                if name in ds.data_vars:
                    ci[name] = ds[name].values.item()
 
            if ci:
                out["conduction_intervals"] = ci
 
        return out

    @staticmethod
    def loadUMCUMedian(
        data_dir: str,
        load: dict,
        pseudo_id: str,
        test_id: str,
        segment_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        rel_path = os.path.join(
            pseudo_id[0:2], pseudo_id[2:4], pseudo_id[4:], f"{test_id}.zarr"
        )
        zarr_path = os.path.join(data_dir, rel_path)

        out = UniversalECGDataset._load_median_core(zarr_path, load, segment_idx)
        out["id_median"] = test_id
        out["filename_median"] = rel_path
        return out

    @staticmethod
    def loadxECGUMCUMedian(
        data_dir: str,
        load: dict,
        pseudo_id: str,
        test_id: str,
        phase_id: str,
        layout: str,
        segment_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        rel_path = os.path.join(
            pseudo_id[0:2], pseudo_id[2:4], pseudo_id[4:], test_id, f"{phase_id}.zarr"
        )
        zarr_path = os.path.join(data_dir, rel_path)

        out = UniversalECGDataset._load_median_core(zarr_path, load, segment_idx)
        out["id_median"] = phase_id
        out["filename_median"] = rel_path
        return out

    @staticmethod
    def loadUniversalMedian(
        data_dir: str,
        load: dict,
        filename: str,
        segment_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        rel_path = f"{filename}.zarr"
        zarr_path = os.path.join(data_dir, rel_path)

        out = UniversalECGDataset._load_median_core(zarr_path, load, segment_idx)
        out["id_median"] = filename
        out["filename_median"] = rel_path
        return out
    
    @staticmethod
    def loadUMCUSample(data_dir: str, pseudo_id: str, test_id: str, segment_idx: Optional[int] =None):
        """
        Loads the raw ECG data stored in the UMCU format into an numpy array.

        ECGs are saved in the numpy format using the following folder structure:
        `waveform_dir/ps/eu/doid/testid.npy`.

        Args:
            data_dir: Directory where the ECGs are stored.
            pseudo_id: The ID denoting the patient or subject.
            test_id: The ID denoting the individual recording.

        Returns:
            A numpy array with the ECG voltage data.
        """
        if segment_idx is not None:
            waveform =  np.load(
                os.path.join(
                    data_dir,
                    pseudo_id[0:2],
                    pseudo_id[2:4],
                    pseudo_id[4:],
                    f"{test_id}.npy",
                ))
            waveform = waveform[:, int(segment_idx*waveform.shape[1]/3):int((segment_idx+1)*waveform.shape[1]/3)]
            return {
            "waveform": waveform,
            "segment_idx": segment_idx,
            "id": test_id,
            "filename": os.path.join(
                pseudo_id[0:2],
                pseudo_id[2:4],
                pseudo_id[4:],
                f"{test_id}.npy",
            ),
        }
        return {
            "waveform": np.load(
                os.path.join(
                    data_dir,
                    pseudo_id[0:2],
                    pseudo_id[2:4],
                    pseudo_id[4:],
                    f"{test_id}.npy",
                )
            ),
            "id": test_id,
            "filename": os.path.join(
                pseudo_id[0:2],
                pseudo_id[2:4],
                pseudo_id[4:],
                f"{test_id}.npy",
            ),
        }

    @staticmethod
    def loadUMCUXECGSample(
        data_dir: str,
        pseudo_id: str,
        test_id: str,
        phase_id: Union[str, list],
        segment_idx: Optional[int] = None,
    ):
        """
        Loads the raw ECG data stored in the UMCU format into a numpy array.

        ECGs are saved in the numpy format using the following folder structure:
        `waveform_dir/ps/eu/doid/testid.npy`.

        Args:
            data_dir: Either a single directory (str) or a dict with keys 'rhythm' and/or 'median'
                    pointing to their respective directories.
            pseudo_id: The ID denoting the patient or subject.
            test_id: The ID denoting the individual recording.
            phase_id: The ID of the ECG phase.

        Returns:
            A dictionary containing any of the available waveforms and their IDs, or None if nothing exists.
        """
        if isinstance(phase_id, list):
            waveform = [
                np.load(
                    os.path.join(
                        data_dir,
                        pseudo_id[0:2],
                        pseudo_id[2:4],
                        pseudo_id[4:],
                        test_id,
                        f"{p}.npy",
                    )
                )
                for p in phase_id
            ]  # shape of [12, 5000, n_phase_id]

            waveform = [w[:, : ((w.shape[1] // 100) * 100)] for w in waveform]
            waveform = np.stack(waveform, axis=-1)

        else:
            waveform = np.load(
                os.path.join(
                    data_dir,
                    pseudo_id[0:2],
                    pseudo_id[2:4],
                    pseudo_id[4:],
                    test_id,
                    f"{phase_id}.npy",
                )
            )
            waveform = waveform[:, : ((waveform.shape[1] // 100) * 100)]
        if waveform.shape[0] > 12:
            waveform = waveform[:12, :]
        return {
            "waveform": waveform,
            "id": phase_id,
            "filename": os.path.join(
                    pseudo_id[0:2],
                    pseudo_id[2:4],
                    pseudo_id[4:],
                    test_id,
                    f"{phase_id}.npy",
                ),
        }

    @staticmethod
    def loadUMCUSampleDig(
        data_dir: str, pseudo_id: str, test_id: str, layout: Optional[str] = None
    ):
        """
        Loads the raw ECG data stored in the UMCU format into an numpy array.

        ECGs are saved in the numpy format using the following folder structure:
        `waveform_dir/ps/eu/doid/testid.npy`.

        Args:
            data_dir: Directory where the ECGs are stored.
            pseudo_id: The ID denoting the patient or subject.
            test_id: The ID denoting the individual recording.

        Returns:
            A numpy array with the ECG voltage data.
        """
        if layout:
            return {
                "waveform": np.load(
                    os.path.join(
                        data_dir,
                        pseudo_id[0:2],
                        pseudo_id[2:4],
                        pseudo_id[4:],
                        layout,
                        f"{test_id}.npy",
                    )
                ),
                "id": test_id,
                "filename": os.path.join(
                    pseudo_id[0:2],
                    pseudo_id[2:4],
                    pseudo_id[4:],
                    layout,
                    f"{test_id}.npy",
                ),
                "layout": layout,
            }
        else:
            return {
                "waveform": np.load(
                    os.path.join(
                        data_dir,
                        pseudo_id[0:2],
                        pseudo_id[2:4],
                        pseudo_id[4:],
                        f"{test_id}.npy",
                    )
                ),
                "id": test_id,
                "filename": os.path.join(
                    pseudo_id[0:2],
                    pseudo_id[2:4],
                    pseudo_id[4:],
                    f"{test_id}.npy",
                ),
            }

    @staticmethod
    def loadUniversalSample(data_dir: str, filename: str):
        """
        Loads the raw ECG data stored in the universal format into an numpy array.

        ECGs are saved in the numpy format using the following folder structure:
        `waveform_dir/filename.npy`.

        Args:
            data_dir: Directory where the ECGs are stored.
            filename: The name of the ECG file.

        Returns:
            A numpy array with the ECG voltage data.
        """
        return {
            "waveform": np.load(
                os.path.join(
                    data_dir,
                    f"{filename}.npy",
                )
            ),
            "id": filename,
            "filename": os.path.join(f"{filename}.npy"),
        }

    @staticmethod
    def loadUniversalHolterSegment(
        data_dir: str,
        filename: str,
        segment_idx: int,
        segment_seconds: int = 10,
    ):
        """
        Loads a single ECG segment from an EDF Holter recording.

        Output:
            waveform:   [12, samples]
            annotation: [samples]
            time:       [samples] (absolute time)
            samplebase: sampling frequency (Hz)
            channels:   list of channel names
            segment_idx: index of the segment in the file
            id:         recording id
            filename:   EDF filename
        """

        path = os.path.join(data_dir, f"{filename}.EDF")

        with open(path, "rb") as f:
            # ---------- Global header ----------
            header = f.read(256).decode("ascii")

            start_date = header[168:176].strip()  # dd.mm.yy
            start_time = header[176:184].strip()  # hh.mm.ss
            start_datetime = datetime.strptime(
                f"{start_date} {start_time}",
                "%d.%m.%y %H.%M.%S",
            )

            n_records = int(header[236:244].strip())
            record_duration = float(header[244:252].strip())
            n_signals = int(header[252:256].strip())

            # ---------- Signal header helpers ----------
            def read_field(width: int) -> list[str]:
                return [f.read(width).decode("ascii").strip() for _ in range(n_signals)]

            # ---------- Signal headers (field-wise, spec-correct) ----------
            labels = read_field(16)
            transducer = read_field(80)
            phys_dims = read_field(8)

            phys_min = np.array(read_field(8), dtype=float)
            phys_max = np.array(read_field(8), dtype=float)
            dig_min = np.array(read_field(8), dtype=int)
            dig_max = np.array(read_field(8), dtype=int)

            prefilter = read_field(80)
            samples_per_record = np.array(read_field(8), dtype=int)
            reserved = read_field(32)

            # ---------- Calibration ----------
            scale = (phys_max - phys_min) / (dig_max - dig_min)
            offset = phys_min - scale * dig_min

            # ---------- Sampling ----------
            fs = samples_per_record[0] / record_duration
            samples_per_segment = int(segment_seconds * fs)

            start_sample = segment_idx * samples_per_segment
            end_sample = start_sample + samples_per_segment

            samples_per_rec = samples_per_record[0]
            record_start = start_sample // samples_per_rec
            record_end = (end_sample + samples_per_rec - 1) // samples_per_rec

            bytes_per_record = samples_per_record.sum() * 2  # int16
            data_start = f.tell()

            f.seek(data_start + record_start * bytes_per_record)

            n_records_to_read = record_end - record_start

            raw = np.frombuffer(
                f.read(n_records_to_read * bytes_per_record),
                dtype="<i2",
            )

        # --- Reshape: [records, signals, samples_per_record] ---
        raw = raw.reshape(n_records_to_read, n_signals, -1)

        # --- Digital -> physical ---
        data = raw * scale[None, :, None] + offset[None, :, None]

        # --- Flatten to sample axis ---
        data = data.transpose(0, 2, 1).reshape(-1, n_signals)

        local_start = start_sample - record_start * samples_per_rec
        local_end = local_start + samples_per_segment

        data = data[local_start:local_end]

        # --- Split ECG / annotation ---
        ecg = data[:, :12].T.astype(np.float32)  # [12, samples]
        annotation = data[:, -1].astype(np.float32)

        # --- Absolute time ---
        offsets = (start_sample + np.arange(samples_per_segment)) / fs
        real_time = np.array(
            [start_datetime + timedelta(seconds=float(t)) for t in offsets],
            dtype="datetime64[s]",
        )
        unique_dims = set(phys_dims[:-1])
        phys_dim = unique_dims.pop() if len(unique_dims) == 1 else None
        return {
            "waveform": ecg,
            "annotation": annotation,
            "time": real_time,
            "samplebase": fs,
            "channels": labels[:12],
            "segment_idx": segment_idx,
            "id": filename,
            "filename": f"{filename}.EDF",
            "gain": (
                0.001
                if phys_dim.strip() == "uV"
                else 1.0 if phys_dim.strip() == "mV" else None
            ),
        }

    @staticmethod
    def loadPhysionetSample(data_dir: str, filename: str):
        """
        Loads the raw ECG data stored in the physionet matlab format into an numpy array.

        ECGs are saved in the numpy format using the following folder structure:
        `waveform_dir/filename.mat`. The raw ECG data is saved in the 'val' variable in Matlab.

        Args:
            data_dir: Directory where the ECGs are stored.
            filename: The name of the ECG file.

        Returns:
            A numpy array with the ECG voltage data.
        """
        path = PhysioNetDataset.split_to_path(filename, steps=3)
        return {
            "waveform": io.loadmat(
                os.path.join(
                    data_dir,
                    path,
                    f"{filename}.mat",
                )
            )["val"],
            "id": filename,
            "filename": os.path.join(
                path,
                f"{filename}.mat",
            ),
        }

    @staticmethod
    def loadPhysionetNumpySample(data_dir: str, filename: str):
        """
        Loads the raw ECG data stored in the physionet numpy format into an numpy array.

        ECGs are saved in the numpy format using the following folder structure:
        `waveform_dir/filename.npy`.

        Args:
            data_dir: Directory where the ECGs are stored.
            filename: The name of the ECG file.

        Returns:
            A numpy array with the ECG voltage data.
        """
        path = PhysioNetDataset.split_to_path(filename, steps=3)
        return {
            "waveform": np.load(os.path.join(data_dir, path, f"{filename}.npy")),
            "filename": os.path.join(path, f"{filename}.npy"),
            "id": filename,
        }

    @staticmethod
    def loadWFDBSample(data_dir: str, filename: str):
        """
        Loads the raw ECG data stored in the physionet numpy format into an numpy array.

        ECGs are saved in the numpy format using the following folder structure:
        `waveform_dir/filename.npy`.

        Args:
            data_dir: Directory where the ECGs are stored.
            filename: The name of the ECG file.

        Returns:
            A numpy array with the ECG voltage data.
        """
        record = wfdb.rdsamp(os.path.join(data_dir, filename))
        return {
            "waveform": record[0].transpose(),
            "samplebase": record[1]["fs"],
            "gain": 1 if record[1]["units"][0] == "mV" else 0.001,
            "filename": os.path.join(filename),
            "id": filename,
        }

    @staticmethod
    def loadHDF5waveform(data_dir, filename):
        with h5py.File(data_dir, "r") as f:
            waveform = f["rhythm"][filename][:]

        return {
            "waveform": waveform,
            "id": filename,
            "filename": os.path.join(data_dir),
        }

    def get_stats(self):
        """
        Calculates size and fraction (percentage) of each label in dataset

        Returns:
            Tuple with a list of dictonaries with
            {'class': [classname], 'size': [num samples in class], 'fraction': [fraction of total dataset]},
            and total dataset size.
        """
        full_size = len(self.dataset.index)
        if not self.labels:
            return [], full_size

        stats = []
        labels = self.labels if isinstance(self.labels, list) else [self.labels]
        for label in labels:
            class_size = len(self.dataset[self.dataset[label] == 1].index)
            stats.append(
                {
                    "class": label,
                    "size": class_size,
                    "fraction": (class_size / full_size) * 100,
                }
            )

        return stats, full_size

    def _load_waveform(self, idx, sample_df_row, segment_idx):
        """
        Internal function to load the ECG waveform from a file in the prespecified format.
        """

        if isinstance(self.dataset_function, Callable):
            # List will be passed through *, meaning the only argument will be the df row
            args = [sample_df_row]
            wf_load_function = self.dataset_function
        else:
            if self.dataset_function == "umcu":
                args = [sample_df_row["PseudoID"], sample_df_row["TestID"], segment_idx]
                wf_load_function = self.loadUMCUSample
                if self.median_beat_dir:
                    median_load_function = self.loadUMCUMedian
            elif self.dataset_function == "umcu_xECG":
                args = [
                    sample_df_row["PseudoID"],
                    sample_df_row["TestID"],
                    sample_df_row["PhaseID"],
                    segment_idx,
                ]
                wf_load_function = self.loadUMCUXECGSample
                if self.median_beat_dir:
                    median_load_function = self.loadxECGUMCUMedian
            elif self.dataset_function == "umcu_dig":
                args = [sample_df_row["PseudoID"], sample_df_row["TestID"]]
                wf_load_function = self.loadUMCUSampleDig

            elif self.dataset_function == "universal":
                args = [sample_df_row["Filename"]]
                wf_load_function = self.loadUniversalSample
                if self.median_beat_dir:
                    median_load_function = self.loadUniversalMedian
            elif self.dataset_function == "universal_holter":
                args = [sample_df_row["Filename"], segment_idx]
                wf_load_function = self.loadUniversalHolterSegment
                if self.median_beat_dir:
                    median_load_function = self.loadUniversalMedian
            elif self.dataset_function == "umcu_sl":
                args = [sample_df_row["PseudoID"], sample_df_row["TestID"], segment_idx]
                wf_load_function = self.loadUMCUSample
                if self.median_beat_dir:
                    median_load_function = self.loadUMCUMedian
            elif self.dataset_function == "physionet":
                args = [sample_df_row["Filename"]]
                wf_load_function = self.loadPhysionetSample

            elif self.dataset_function == "physionet_numpy":
                args = [sample_df_row["Filename"]]
                wf_load_function = self.loadPhysionetNumpySample

            elif self.dataset_function == "wfdb":
                args = [sample_df_row["Filename"]]
                wf_load_function = self.loadWFDBSample

            elif self.dataset_function == "hdf5":
                args = [sample_df_row["Filename"]]
                wf_load_function = self.loadHDF5waveform

        sample = wf_load_function(self.waveform_dir, *args)

        if self.median_beat_dir:
            median_sample = median_load_function(
                self.median_beat_dir, self.median_load_params, *args
            )
            sample.update(median_sample)

        if self.secondary_waveform_dir:
            secondary_sample = wf_load_function(self.secondary_waveform_dir, *args)

            if self.secondary_median_beat_dir:
                secondary_median_sample = median_load_function(
                    self.secondary_median_beat_dir, self.median_load_params, *args
                )
                secondary_sample.update(secondary_median_sample)

        else:
            secondary_sample = None

        return sample, secondary_sample

    def __getitem__(self, idx: int) -> dict:
        file_idx, segment_idx = self.index_map[idx]
        sample_df_row = self.dataset.iloc[file_idx]
        try:
            sample, secondary_waveform = self._load_waveform(
                idx, sample_df_row, segment_idx
            )
        except Exception as e:
            print(e)
            print(
                "Above error was caught in dataloader, returning neighbouring sample to continue training"
            )
            idx = idx - 1 if idx > 0 else idx + 1
            return self.__getitem__(idx)
        sample["attrs"] = {"log_image": False}
        if idx in self.log_images:
            sample["attrs"]["log_image"] = True

        # Add waveform, original sample base, gain and ID to sample

        if (
            "samplebase" not in sample
            and "SampleBase" in sample_df_row
            and not pd.isna(sample_df_row["SampleBase"])
        ):
            sample["samplebase"] = int(sample_df_row["SampleBase"])

        if (
            "gain" not in sample
            and "Gain" in sample_df_row
            and not pd.isna(sample_df_row["Gain"])
        ):
            sample["gain"] = float(sample_df_row["Gain"])

        sample["data_idx"] = idx

        if secondary_waveform is not None:
            for i in secondary_waveform.keys():
                if i not in ["id"]:
                    sample[f"secondary_{i}"] = secondary_waveform[i]

        # Sometimes additional information is needed (e.g. for a median cutoff)
        possible_cols = [
            "AcqDate",
            # "POnset",
            # "TOffset",
            # "VentricularRate",
            # "QOnset",
            # "POffset",
            # "QOffset",
            # "start_idx",
            # "end_idx",
            # "Lead",
            "Layout",
            "Gender",
            "Age",
            "DateofBirth",
            "HeartRate",
            "Load (W)",
            "Load (%)",
            "Speed (rpm)",
            "Speed (km/h)",
            "HeartRate",
            "AcquisitionTime",
            "AcquisitionDate",
            "AcquisitionDateTime",
            "PseudoID",
            "TestID",
            "PhaseID",
            "Phase",
            "ST-Slope",
            "J-Point",
            "PseudoID",
            "TestID",
            "fname",
        ] + [f"TrueBaseline_{i}" for i in range(12)]

        for col in possible_cols:
            if col in self.dataset:
                sample[col.lower()] = sample_df_row[col]

        if self.additional_dataset_function:
            args = [sample_df_row]
            for func in self.additional_dataset_function:
                sample.update(func(self, *args))

        if self.labels:
            if isinstance(self.labels, list):
                labels = self.label_set.iloc[idx]
                sample["label"] = torch.from_numpy(labels.values.astype("int64"))
            elif self.labels in sample:
                # Additional function already processed this label column
                sample["label"] = sample[self.labels]
            elif self.labels in self.dataset.columns.values:
                # Fallback to raw column value
                sample["label"] = sample_df_row[self.labels]

        try:
            if self.transform:
                # for now always applies the same transforms to secondary sample
                sample = self.transform(sample)
        except Exception as e:
            print(f"Error in sample idx={idx}: {e}")
            # Try up to 3 neighboring samples before giving up
            for attempt in range(1, 4):
                neighbor_idx = idx - attempt if idx - attempt >= 0 else idx + attempt
                if neighbor_idx >= len(self.dataset):
                    continue
                try:
                    return self.__getitem__(neighbor_idx)
                except Exception as e2:
                    print(f"Retry {attempt} failed at idx={neighbor_idx}: {e2}")
                    continue
            print(f"Skipping idx={idx} after multiple failures.")
        return sample

    def get_item_on_id_(self, col, id):
        "Internal function to return an ECG sample."
        idx = self.dataset.index[self.dataset[col] == id].to_list()[0]
        sample = self.__getitem__(idx)
        return sample

    def read_holter_metadata(self, path: str, segment_seconds: int):
        """
        Read EDF holter metadata and compute how many fixed-length segments
        fit in the recording.

        Returns
        -------
        n_segments : int
            Number of full segments
        start_time : np.datetime64
            Absolute start time of the recording
        end_time : np.datetime64
            Absolute end time of the recording
        """

        # ---------- Read EDF header directly (date & time) ----------
        with open(path, "rb") as f:
            header = f.read(256).decode("ascii")

            start_date = header[168:176].strip()  # dd.mm.yy
            start_time_str = header[176:184].strip()  # hh.mm.ss

            start_datetime = datetime.strptime(
                f"{start_date} {start_time_str}",
                "%d.%m.%y %H.%M.%S",
            )

        start_time = np.datetime64(start_datetime, "s")

        # ---------- Read sampling metadata ----------
        with pyedflib.EdfReader(path) as f:
            n_records = f.datarecords_in_file
            record_duration = f.datarecord_duration
            n_signals = f.signals_in_file

            samples_per_record = np.array(
                [f.samples_in_datarecord(i) for i in range(n_signals)],
                dtype=np.int64,
            )

        fs = samples_per_record[0] / record_duration
        total_samples = int(n_records * samples_per_record[0])

        samples_per_segment = int(segment_seconds * fs)
        n_segments = total_samples // samples_per_segment

        total_duration_seconds = total_samples / fs
        end_time = start_time + np.timedelta64(int(total_duration_seconds), "s")

        return n_segments, start_time, end_time

    def __len__(self) -> int:
        return len(self.index_map)

    def __gettime__(self, idx):
        return self.file_time_info[idx]


class PhysioNetDataset(UniversalECGDataset):
    @staticmethod
    def split_to_path(filename, steps=3):
        """
        Generates path a path where every 2 characters from the input filename form a directory
        """
        return os.path.join(*[filename[i : i + 2] for i in range(steps)])

    # TODO make property or constant somewhere else
    @staticmethod
    def SNOMEDCT_To_Abbreviation_map():
        return {
            "164889003": "AF",
            "164890007": "AFL",
            "6374002": "BBB",
            "426627000": "Brady",
            "733534002": "CLBBB",
            "713427006": "CRBBB",
            "270492004": "IAVB",
            "713426002": "IRBBB",
            "39732003": "LAD",
            "445118002": "LAnFB",
            "164909002": "LBBB",
            "251146004": "LQRSV",
            "698252002": "NSIVCB",
            "426783006": "NSR",
            "284470004": "PAC",
            "10370003": "PR",
            "365413008": "PRWP",
            "427172004": "PVC",
            "164947007": "LPR",
            "111975006": "LQT",
            "164917005": "QAb",
            "47665007": "RAD",
            "59118001": "RBBB",
            "427393009": "SA",
            "426177001": "SB",
            "427084000": "STach",
            "63593006": "SVPB",
            "164934002": "TAb",
            "59931005": "TInv",
            "17338001": "VPB",
        }

    @staticmethod
    def extract_header_info(file_content, file_id):
        _, n_leads, sample_base, num_samples, date, time = file_content[0].split()

        info = {
            "Filename": file_id,
            "n_leads": int(n_leads),
            "SampleBase": int(sample_base),
            "NumSamples": int(num_samples),
            "Date": date,
            "Time": time,
            "AF": 0,
            "AFL": 0,
            "BBB": 0,
            "Brady": 0,
            "CLBBB": 0,
            "CRBBB": 0,
            "IAVB": 0,
            "IRBBB": 0,
            "LAD": 0,
            "LAnFB": 0,
            "LBBB": 0,
            "LQRSV": 0,
            "NSIVCB": 0,
            "NSR": 0,
            "PAC": 0,
            "PR": 0,
            "PRWP": 0,
            "PVC": 0,
            "LPR": 0,
            "LQT": 0,
            "QAb": 0,
            "RAD": 0,
            "RBBB": 0,
            "SA": 0,
            "SB": 0,
            "STach": 0,
            "SVPB": 0,
            "TAb": 0,
            "TInv": 0,
            "VPB": 0,
        }

        if n_leads != "12":
            raise RuntimeError(f"Found non 12-lead entry, {file_id}")

        sr = set()
        baseline_correction = {}
        extra_fields = ["#Rx:", "#Hx:", "#Sx:"]
        try:
            for idx, l in enumerate(file_content[1:]):
                l = l.split()
                if l[0][-4:] == ".mat":
                    baseline_correction[f"TrueBaseline_{idx}"] = int(l[4])
                    sr.add(l[2])
                elif l[0] == "#Age:":
                    info["Age"] = float(l[-1])
                elif l[0] == "#Sex:":
                    info["Sex"] = PhysioNetDataset.convert_sex(l[-1])
                elif l[0] == "#Dx:":
                    for diag in l[-1].split(","):
                        if (
                            diag
                            in PhysioNetDataset.SNOMEDCT_To_Abbreviation_map().keys()
                        ):
                            diag = PhysioNetDataset.SNOMEDCT_To_Abbreviation_map()[diag]
                            info[diag] = 1
                elif l[0] in extra_fields:
                    info[l[0][1:]] = l[-1]

            if len(sr) > 1:
                raise RuntimeError(f"Found inconsistend gain per lead in {file_id}")

            info.update(baseline_correction)
            info["Gain"] = float(sr.pop().split("/")[0]) * 1e-6

        except Exception as e:
            print(file_content)
            raise e

        return info

    @staticmethod
    def convert_sex(sex):
        if sex in ("Female", "female", "F", "f"):
            return 0
        elif sex in ("Male", "male", "M", "m"):
            return 1
        return float("nan")

    def convert_to_numpy(self, df, path):
        print(" -- Performing numpy conversion...")
        for filename in tqdm(df["Filename"]):
            file_path = PhysioNetDataset.split_to_path(filename, steps=3)
            if not os.path.exists(os.path.join(path, file_path, f"{filename}.npy")):
                data = io.loadmat(
                    os.path.join(
                        path,
                        file_path,
                        f"{filename}.mat",
                    )
                )["val"]
                np.save(os.path.join(path, file_path, f"{filename}.npy"), data)
        print("Finished numpy conversion")

    def extract_archive(
        self,
        remove_finished: bool = False,
    ):
        if self.to_path is None:
            self.to_path = os.path.dirname(self.from_path)

        print("Now extracting data from tar archive, this may take a while...")

        process = subprocess.Popen(
            f"tar zxf {self.from_path} -C {self.to_path}", shell=True
        )
        process.wait()

        print(f"Done extracting - exit code = {process.returncode}")

        print("Now extracting header info and moving files")
        header_info = []

        for file in os.listdir(f"{self.to_path}/{self.archive_root}"):
            full_file_path = f"{self.to_path}/{self.archive_root}/{file}"
            file_id, ftype = file.split(".")

            # Only read header files, extract raw data in header file is found
            if ftype == "hea":
                with open(full_file_path, "r") as extracted:
                    # Get header info from header file
                    header_info.append(
                        self.extract_header_info(extracted.readlines(), file_id)
                    )

                mat_file_name = f"{self.to_path}/{self.archive_root}/{file_id}.mat"

                # Create folder structure for raw data
                file_path = PhysioNetDataset.split_to_path(file_id, steps=3)
                f_to_path = os.path.join(self.to_path, file_path)

                # Move files
                os.renames(mat_file_name, os.path.join(f_to_path, f"{file_id}.mat"))
                os.renames(full_file_path, os.path.join(f_to_path, f"{file_id}.hea"))

        header_df = pd.DataFrame(header_info)
        header_df.to_csv(f"{self.to_path}/header_info.csv")

        print("All done")

        if remove_finished:
            os.remove(self.from_path)


class PTBXLDataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./PTB_XL",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_PTBXL.tar.gz",
                path,
                filename="PTB_XL.tar.gz",
                md5="55e8a5c25eadfeff4fcd38f5bbf3cb13",
            )

        self.archive_root = "WFDB_PTBXL"
        self.from_path = f"{path}/PTB_XL.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(PTBXLDataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class CPSC2018Dataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./CPSC_2018",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_CPSC2018.tar.gz",
                path,
                filename="CPSC_2018.tar.gz",
                md5="5d4b520e3b6558a33dc9dbe49d08f8f1",
            )

        self.archive_root = "WFDB_CPSC2018"
        self.from_path = f"{path}/CPSC_2018.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(CPSC2018Dataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class China12LeadDataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./China_12_Lead",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_CPSC2018_2.tar.gz",
                path,
                filename="China_12_Lead.tar.gz",
                md5="5b1498abacaa1b5a762691c006e737ad",
            )

        self.archive_root = "WFDB_CPSC2018_2"
        self.from_path = f"{path}/China_12_Lead.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(China12LeadDataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class INCARTDataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./INCART",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_StPetersburg.tar.gz",
                path,
                filename="INCART.tar.gz",
                md5="525dde6bd26bff0dcb35189e78ae7d6d",
            )

        self.archive_root = "WFDB_StPetersburg"
        self.from_path = f"{path}/INCART.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(INCARTDataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class PTBDataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./PTB",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_PTB.tar.gz",
                path,
                filename="PTB.tar.gz",
                md5="3df4662a8a9189a6a5924424b0fcde0e",
            )

        self.archive_root = "WFDB_PTB"
        self.from_path = f"{path}/PTB.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(PTBDataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class Georgia12LeadDataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./Georgia_12_Lead",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_Ga.tar.gz",
                path,
                filename="Georgia_12_Lead.tar.gz",
                md5="47085dd62baca5ace4041025d6910b13",
            )

        self.archive_root = "WFDB_Ga"
        self.from_path = f"{path}/Georgia_12_Lead.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(Georgia12LeadDataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class ChapmanUniversityDataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./Chapman_University",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_ChapmanShaoxing.tar.gz",
                path,
                filename="Chapman_University.tar.gz",
                md5="a3e10171eba1e7520a38919594d834e5",
            )

        self.archive_root = "WFDB_ChapmanShaoxing"
        self.from_path = f"{path}/Chapman_University.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(ChapmanUniversityDataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class NingboDataset(PhysioNetDataset):
    def __init__(
        self,
        path: str = "./Ningbo",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_Ningbo.tar.gz",
                path,
                filename="Ningbo.tar.gz",
                md5="84171145922078146875394acb89b765",
            )

        self.archive_root = "WFDB_Ningbo"
        self.from_path = f"{path}/Ningbo.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(NingboDataset, self).__init__(
            dataset_function="physionet_numpy" if use_numpy else "physionet",
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )


class LUDBDataset(PhysioNetDataset):
    @staticmethod
    def load_LUDB_annotation(dataset_obj: UniversalECGDataset, df_row):
        file_path = PhysioNetDataset.split_to_path(str(df_row["ID"]), steps=2)
        return {
            "annotation": np.load(
                os.path.join(
                    dataset_obj.waveform_dir,
                    "annot_npy",
                    file_path,
                    f'{df_row["ID"]}.npy',
                ),
                allow_pickle=True,
            )
        }

    @staticmethod
    def load_LUDB_waveform_npy(waveform_dir: str, df_row):
        file_path = PhysioNetDataset.split_to_path(str(df_row["ID"]), steps=2)
        return np.load(
            os.path.join(waveform_dir, "data_npy", file_path, f'{df_row["ID"]}.npy')
        ), str(df_row["ID"])

    @staticmethod
    def load_LUDB_waveform_WFDB(waveform_dir: str, df_row):
        signal, _ = wfdb.rdsamp(os.path.join(waveform_dir, "data", df_row["ID"])), str(
            df_row["ID"]
        )
        return signal

    def __init__(
        self,
        path: str = "./LuDB",
        download: bool = True,
        use_numpy: bool = True,
        *args,
        **kwargs,
    ):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                "https://physionet.org/static/published-projects/ludb/lobachevsky-university-electrocardiography-database-1.0.1.zip",
                path,
                filename="LuDB.tar.gz",
            )

        self.archive_root = "lobachevsky-university-electrocardiography-database-1.0.1"
        self.from_path = f"{path}/LuDB.tar.gz"
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough?
        if not os.path.exists(f"{path}/ludb.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/ludb.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        if "additional_dataset_function" in kwargs:
            kwargs["additional_dataset_function"] = [
                LUDBDataset.load_LUDB_annotation
            ] + kwargs["additional_dataset_function"]
        else:
            kwargs["additional_dataset_function"] = [LUDBDataset.load_LUDB_annotation]

        super(LUDBDataset, self).__init__(
            dataset_function=(
                LUDBDataset.load_LUDB_waveform_npy
                if use_numpy
                else LUDBDataset.load_LUDB_waveform_WFDB
            ),
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs,
        )

    @staticmethod
    def possible_leads():
        return [
            "i",
            "ii",
            "iii",
            "avr",
            "avl",
            "avf",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
        ]

    def convert_to_numpy(self, df, path):
        print(" -- Performing numpy conversion...")

        for filename in tqdm(df["Filename"]):
            filename = filename.split(".")[0]
            file_path = PhysioNetDataset.split_to_path(filename, steps=2)

            if not os.path.exists(
                os.path.join(path, "data_npy", file_path, f"{filename}.npy")
            ) or not os.path.exists(
                os.path.join(path, "annot_npy", file_path, f"{filename}.npy")
            ):
                signal, meta = wfdb.rdsamp(os.path.join(path, "data", filename))

                if not os.path.exists(os.path.join(path, "data_npy", file_path)):
                    os.makedirs(os.path.join(path, "data_npy", file_path))

                np.save(
                    os.path.join(path, "data_npy", file_path, f"{filename}.npy"),
                    signal.transpose(),
                )

                if not os.path.exists(os.path.join(path, "annot_npy", file_path)):
                    os.makedirs(os.path.join(path, "annot_npy", file_path))

                annotations = np.zeros((12, 5000), object)
                annotations[:] = ""
                for lead_idx, lead in enumerate(LUDBDataset.possible_leads()):
                    ann = wfdb.rdann(os.path.join(path, "data", filename), lead)
                    onset = 0

                    symbols = list(zip(ann.sample, ann.symbol))
                    for pair_idx, (symbol_idx, symbol) in enumerate(symbols):
                        if symbol == "(":
                            # Skip t onset and leave it at qrs offset
                            if (
                                len(symbols) > (pair_idx - 1)
                                and symbols[pair_idx + 1][1] == "t"
                            ):
                                continue

                            onset = symbol_idx
                        elif symbol in ["p", "N", "t"]:
                            last_peak = symbol
                            last_peak_idx = symbol_idx
                        elif symbol == ")":
                            annotations[lead_idx, onset:symbol_idx] = last_peak
                            annotations[lead_idx, last_peak_idx] = last_peak + "_PEAK"
                            onset = symbol_idx

                np.save(
                    os.path.join(path, "annot_npy", file_path, f"{filename}.npy"),
                    annotations,
                )
        print("Finished numpy conversion")

    def extract_archive(
        self,
        remove_finished: bool = False,
    ):
        if self.to_path is None:
            self.to_path = os.path.dirname(self.from_path)

        print("Now extracting data from tar archive, this may take a while...")

        process = subprocess.Popen(
            f"unzip {self.from_path} -d {self.to_path}", shell=True
        )
        process.wait()

        print(f"Done extracting - exit code = {process.returncode}")

        print("Now moving files")

        for file in os.listdir(f"{self.to_path}/{self.archive_root}"):
            full_file_path = f"{self.to_path}/{self.archive_root}/{file}"
            os.renames(full_file_path, os.path.join(self.to_path, file))

        df = pd.read_csv(f"{self.to_path}/ludb.csv")
        df["SampleBase"] = 500
        df["Gain"] = 0.001
        df["Filename"] = df["ID"].astype("string") + ".dat"
        df.to_csv(f"{self.to_path}/ludb.csv")


class DatasetPaired(Dataset):
    def __init__(
        self,
        dataset: UniversalECGDataset,
        pair_on: list,
        num_bins: int,
        constrastive_pairs: bool = False,
    ):

        self.pair_on = pair_on
        self.dataset = dataset
        self.dataset_df = dataset.dataset
        self.num_bins = num_bins
        self.constrastive_pairs = constrastive_pairs

        self.binned_dfs = defaultdict(list)
        self.bin_borders = {}
        self.idx_list = np.arange(len(dataset))

        for p_on in pair_on:
            assert p_on in dataset.dataset, f"Could not find '{p_on}' in the dataset"

            max_uniq = len(dataset.dataset[p_on].unique())
            if max_uniq < num_bins:
                warnings.warn(
                    f"Variable '{p_on}' has less unique values ({max_uniq}) then number of bins ({num_bins}), using {max_uniq} discrete bins instead"
                )
                self._cutDiscreteBins(p_on)
            else:
                self._cutBins(p_on)

    def _cutBins(self, p_on):
        bins, borders = pd.qcut(
            self.dataset.dataset[p_on],
            self.num_bins,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        self.bin_borders[p_on] = {"discrete": False, "borders": borders}
        for bin_idx in range(len(borders)):
            self.binned_dfs[p_on].append(self.idx_list[bins == bin_idx])

    def _cutDiscreteBins(self, p_on):
        self.bin_borders[p_on] = {
            "discrete": True,
            "borders": self.dataset.dataset[p_on].unique(),
        }
        for discrete_bin in self.bin_borders[p_on]["borders"]:
            # TODO floating point precision?
            self.binned_dfs[p_on].append(
                self.idx_list[(self.dataset.dataset[p_on] == discrete_bin)]
            )

    def _get_sample_bins(self, idx):
        df_row = self.dataset_df.iloc[idx]

        cur_sample_bins = {}

        for p_on in self.pair_on:
            borders = self.bin_borders[p_on]["borders"]
            is_discrete = self.bin_borders[p_on]["discrete"]

            if is_discrete:
                # TODO floating point precision?
                cur_sample_bins[p_on] = (borders == df_row[p_on]).argmax()
            else:
                bin_onsets = borders[:-1]
                bin_offsets = borders[1:]

                for bin_idx, (onset, offset) in enumerate(zip(bin_onsets, bin_offsets)):
                    if df_row[p_on] >= onset and df_row[p_on] <= offset:
                        cur_sample_bins[p_on] = bin_idx

        cur_sample_bins["residual"] = float("nan")

        return cur_sample_bins

    def __getitem__(self, idx, fixed_pair_on=None):
        sample = self.dataset.__getitem__(idx)
        cur_sample_bins = self._get_sample_bins(idx)

        # TODO move up and prevent loop?
        if not fixed_pair_on:
            rand_pair_on_idx = np.random.randint(0, len(self.pair_on))
            rand_pair_on = self.pair_on[rand_pair_on_idx]
        else:
            rand_pair_on = fixed_pair_on

        if not self.constrastive_pairs:
            bin_oi = cur_sample_bins[rand_pair_on]
        else:
            choices = np.arange(len(self.bin_borders[rand_pair_on]) - 1)
            choices.pop(cur_sample_bins[rand_pair_on])
            bin_oi = np.random.choice(choices)

        sample2_idx = np.random.choice(self.binned_dfs[rand_pair_on][bin_oi])
        sample2 = self.dataset.__getitem__(sample2_idx)
        sample_2_bins = self._get_sample_bins(sample2_idx)
        factor_mask = np.array(list(cur_sample_bins.values())) == np.array(
            list(sample_2_bins.values())
        )

        return {
            "sample1": sample,
            "sample2": sample2,
            "factor": rand_pair_on_idx,
            "factor_mask": factor_mask,
        }

    # def get_dataframe(self):
    #     """
    #     Returns the dataframe that is being used in this dataset.
    #     """
    #     return self.dataset

    def query_dataset(
        self,
        query: List[tuple],
        max_num_samples: int = None,
        return_indices: bool = False,
    ):
        """
        Query dataset for specific class (combinations), returns samples (dict with wavefrom etc) or indices in dataset.

        Args:
            query: Defines what classes and bins returned samples should (not) have e.g. [('rhythm_st', 3, True), ('conduction_lbtb', 0, False)]
            max_num_samples: Maximum number of samples to return from query (default: {None})
            return_indices: Wether to return samples (dict with waveform etc) or indices in dataset (default: {False})

        Returns:
            List of samples or list of indices that meet the query.
        """
        query = query if type(query) == list else [query]

        possible_samples = np.zeros(len(self.dataset_df.index), dtype=bool)
        impossible_samples = np.zeros(len(self.dataset_df.index), dtype=bool)

        for label, bin_idx, tf in query:
            assert not (
                bin_idx > (len(self.binned_dfs[label]))
            ), f"Bin idx for query ('{label}', {bin_idx}, {tf}) is out of bounds"
            bin_oi = self.binned_dfs[label][bin_idx]
            if tf:
                possible_samples[bin_oi] = True
            else:
                impossible_samples[bin_oi] = True

        possible_samples = np.arange(len(self.dataset_df.index))[
            possible_samples & (~impossible_samples)
        ]

        if max_num_samples and max_num_samples < possible_samples.shape[0]:
            possible_samples = possible_samples[:max_num_samples]

        if return_indices:
            return possible_samples

        return [self.__getitem__(qidx) for qidx in possible_samples]

    def get_stats(self):
        """
        Calculates size and fraction (percentage) of each label and bin in dataset

        Returns:
            Tuple with a list of dictonaries with
            {'class': [classname], 'size': [num samples in class], 'fraction': [fraction of total dataset]},
            and total dataset size.
        """
        full_size = len(self.dataset_df.index)

        stats = []
        for label in self.binned_dfs.keys():
            bins = self.binned_dfs[label]
            bin_borders = self.bin_borders[label]["borders"]

            for bin_idx, (c_bin, c_border) in enumerate(zip(bins, bin_borders)):
                class_size = len(c_bin)
                stats.append(
                    {
                        "class": f"{label} [{bin_idx}] ({c_border})",
                        "size": class_size,
                        "fraction": (class_size / full_size) * 100,
                    }
                )

        return stats, full_size

    def print_stats(self):
        """Prints statistics of the dataset."""
        stats, full_size = self.get_stats()
        print("-- Dataset distribution -- ")
        print(f"Full size: {full_size}")
        for s in stats:
            print(f'["{s["class"]}"] - Num entries: {s["size"]} ({s["fraction"]:.3}%)')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)
