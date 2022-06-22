"""
ECG datasets to use with ECGx.AI package.

MB Vessies and RR van de Leur
"""

import os
import torch
from torch.utils.data import Dataset
from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Union, Callable
import pandas as pd
from torchvision.transforms import Compose
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import subprocess


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
        query : List[tuple],
        max_num_samples : int = None,
        return_indices : bool = False
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

        qs = " and ".join([f'{param} == {val}' for (param, val) in query])
        qres = self.dataset.query(qs)
        if max_num_samples and max_num_samples < qres.shape[0]:
            qres = qres[:max_num_samples]
        
        if return_indices:
            return [self.dataset.index.get_loc(qidx) for qidx in qres.index]

        return [self.__getitem__(self.dataset.index.get_loc(qidx)) for qidx in qres.index]

    def print_stats(self):
        """ Prints statistics of the dataset. """
        stats, full_size = self.get_stats()
        print('-- Dataset distribution -- ')
        print(f'Full size: {full_size}')
        for s in stats:
            print(f'["{s["class"]}"] - Num entries: {s["size"]} ({s["fraction"]:.3}%)')

    def __len__(self):
        """ Returns the length of the dataset. """
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
                options are 'umcu', 'universal', 'physionet' and 'physionet_numpy'. In the 
                UMCU datasets, ECGs are saved in the numpy format using the following folder structure:
                `waveform_dir/ps/eu/doid/testid.npy`. In the universal datasets, files are also saved using
                the numpy format, but then all in one folder: `waveform_dir/filename.npy`. For the physionet
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
    """
    def __init__(
        self, 
        dataset_function: Union[str, Callable],
        waveform_dir: str,
        dataset: pd.DataFrame,
        transform: Union[Compose, None] = None,
        labels: Union[str, List[str]] = 'Label',
        secondary_waveform_dir: str = ""
    ):
        """
        Initializes the ECG datasets.
        """
        assert dataset_function in ['umcu', 'universal', 'physionet', 'physionet_numpy'] or isinstance(dataset_function, Callable)

        if dataset_function == "umcu":
            if not ('PseudoID' in dataset and 'TestID' in dataset):
                raise ValueError("Please provide a dataframe with both PseudoID and TestID columns when using the UMCU format.")
        elif dataset_function in ['universal', 'physionet', 'physionet_numpy']:
            if 'Filename' not in dataset:
                raise ValueError("Please provide a dataframe with Filename column when using this format.")

        self.dataset = dataset
        self.waveform_dir = waveform_dir
        self.secondary_waveform_dir = secondary_waveform_dir
        self.transform = transform
        self.labels = labels
        self.label_indices = [self.dataset.columns.get_loc(c) for c in labels] if isinstance(labels, list) else None
        self.dataset_function = dataset_function

    def train_test_split(self, ratio=0.1, shuffle=True):
        """
            Splits the current dataset into a train and testset. 
        """
        trainset_df, testset_df = train_test_split(self.dataset, test_size=ratio, shuffle=shuffle)

        trainset = UniversalECGDataset(
            self.dataset_function,
            self.waveform_dir,
            trainset_df,
            self.transform,
            self.labels,
            self.secondary_waveform_dir,
        )

        testset = UniversalECGDataset(
            self.dataset_function,
            self.waveform_dir,
            testset_df,
            self.transform,
            self.labels,
            self.secondary_waveform_dir,
        )

        return trainset, testset

    @staticmethod
    def loadUMCUSample(data_dir: str, pseudo_id: str, test_id: str):
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
        return np.load(os.path.join(data_dir, pseudo_id[0:2], pseudo_id[2:4], pseudo_id[4:], f'{test_id}.npy',)), test_id

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
        return np.load(os.path.join(data_dir, f'{filename}.npy',)), filename

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
        return io.loadmat(os.path.join(data_dir, path, f'{filename}.mat',))['val'], filename

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
        return np.load(os.path.join(data_dir, path, f'{filename}.npy')), filename

    def get_stats(self):
        """
        Calculates size and fraction (percentage) of each label in dataset 

        Returns:
            Tuple with a list of dictonaries with 
            {'class': [classname], 'size': [num samples in class], 'fraction': [fraction of total dataset]},
            and total dataset size.
        """
        full_size = len(self.dataset.index)
        stats = []
        labels = self.labels if isinstance(self.labels, list) else [self.labels]
        for label in labels:
            class_size = len(self.dataset[self.dataset[label] == 1].index)
            stats.append({'class': label, 'size': class_size, 'fraction': (class_size / full_size) * 100})

        return stats, full_size

    def _load_waveform(self, idx):
        """
        Internal function to load the ECG waveform from a file in the prespecified format.
        """
        secondary_waveform = None
        if isinstance(self.dataset_function, Callable):
            # List will be passed through *, meaning the only argument will be the df row
            args = [self.dataset.iloc[idx]]
            wf_load_function = self.dataset_function
        else:
            if self.dataset_function == 'umcu':
                args = [self.dataset['PseudoID'].iloc[idx], self.dataset['TestID'].iloc[idx]]
                wf_load_function = self.loadUMCUSample

            elif self.dataset_function == 'universal':
                args = [self.dataset['Filename'].iloc[idx]]
                wf_load_function = self.loadUniversalSample

            elif self.dataset_function == 'physionet':
                args = [self.dataset['Filename'].iloc[idx]]
                wf_load_function = self.loadPhysionetSample

            elif self.dataset_function == 'physionet_numpy':
                args = [self.dataset['Filename'].iloc[idx]]
                wf_load_function = self.loadPhysionetNumpySample

        waveform, sample_id = wf_load_function(self.waveform_dir, *args)
        if self.secondary_waveform_dir:
            secondary_waveform, _ = wf_load_function(self.secondary_waveform_dir, *args)

        return sample_id, waveform, secondary_waveform

    def __getitem__(self, idx):
        "Internal function to return an ECG sample."
        sample_id, waveform, secondary_waveform = self._load_waveform(idx)

        # Add waveform, original sample base, gain and ID to sample
        sample = {
            'waveform': waveform,
            'samplebase': int(self.dataset['SampleBase'].iloc[idx]),
            'gain': float(self.dataset['Gain'].iloc[idx]),
            'id': sample_id,
        }

        if secondary_waveform is not None:
            sample['secondary_waveform'] = secondary_waveform

        # Sometimes additional information is needed (e.g. for a median cutoff)
        possible_cols = ['AcqDate', 'POnset', 'TOffset', 'VentricularRate',
                         'QOnset', 'POffset', 'QOffset', 'start_idx',
                         'end_idx'] + [f'TrueBaseline_{i}' for i in range(12)]

        for col in possible_cols:
            if col in self.dataset:
                sample[col.lower()] = self.dataset[col].iloc[idx]

        if self.labels:
            if isinstance(self.labels, list):
                labels = self.dataset.iloc[idx, self.label_indices].astype('int64')
                sample['label'] = torch.from_numpy(labels.values)
            elif self.labels in self.dataset.columns.values:
                label = self.dataset[self.labels].iloc[idx]
                sample['label'] = label

        try:
            if self.transform:
                # for now always applies the same transforms to secondary sample
                sample = self.transform(sample)
        except Exception as e:
            print(e)
            print("Above error was caught in dataloader, returning neighbouring sample to continue training")
            idx = idx - 1 if idx > 0 else idx + 1
            return self.__getitem__(idx)

        return sample


class PhysioNetDataset(UniversalECGDataset):
    @staticmethod
    def split_to_path(filename, steps=3):
        """
         Generates path a path where every 2 characters from the input filename form a directory
        """
        return os.path.join(*[filename[i:i + 2] for i in range(steps)])

    # TODO make property or constant somewhere else
    @ staticmethod
    def SNOMEDCT_To_Abbreviation_map():
        return {'164889003': 'AF', '164890007': 'AFL', '6374002': 'BBB', '426627000': 'Brady', '733534002': 'CLBBB', '713427006': 'CRBBB', '270492004': 'IAVB', '713426002': 'IRBBB', '39732003': 'LAD', '445118002': 'LAnFB', '164909002': 'LBBB', '251146004': 'LQRSV', '698252002': 'NSIVCB', '426783006': 'NSR', '284470004': 'PAC', '10370003': 'PR', '365413008': 'PRWP', '427172004': 'PVC', '164947007': 'LPR', '111975006': 'LQT', '164917005': 'QAb', '47665007': 'RAD', '59118001': 'RBBB', '427393009': 'SA', '426177001': 'SB', '427084000': 'STach', '63593006': 'SVPB', '164934002': 'TAb', '59931005': 'TInv', '17338001': 'VPB'}

    @ staticmethod
    def extract_header_info(file_content, file_id):
        _, n_leads, sample_base, num_samples, date, time = file_content[0].split()

        info = {
            'Filename': file_id,
            'n_leads': int(n_leads),
            'SampleBase': int(sample_base),
            'NumSamples': int(num_samples),
            'Date': date,
            'Time': time,
            'AF': 0,
            'AFL': 0,
            'BBB': 0,
            'Brady': 0,
            'CLBBB': 0,
            'CRBBB': 0,
            'IAVB': 0,
            'IRBBB': 0,
            'LAD': 0,
            'LAnFB': 0,
            'LBBB': 0,
            'LQRSV': 0,
            'NSIVCB': 0,
            'NSR': 0,
            'PAC': 0,
            'PR': 0,
            'PRWP': 0,
            'PVC': 0,
            'LPR': 0,
            'LQT': 0,
            'QAb': 0,
            'RAD': 0,
            'RBBB': 0,
            'SA': 0,
            'SB': 0,
            'STach': 0,
            'SVPB': 0,
            'TAb': 0,
            'TInv': 0,
            'VPB': 0
        }

        if n_leads != '12':
            raise RuntimeError(f"Found non 12-lead entry, {file_id}")

        sr = set()
        baseline_correction = {}
        extra_fields = ['#Rx:', '#Hx:', '#Sx:']
        try:
            for idx, l in enumerate(file_content[1:]):
                l = l.split()
                if l[0][-4:] == '.mat':
                    baseline_correction[f'TrueBaseline_{idx}'] = int(l[4])
                    sr.add(l[2])
                elif l[0] == '#Age:':
                    info['Age'] = float(l[-1])
                elif l[0] == '#Sex:':
                    info['Sex'] = PhysioNetDataset.convert_sex(l[-1])
                elif l[0] == '#Dx:':
                    for diag in l[-1].split(','):
                        if diag in PhysioNetDataset.SNOMEDCT_To_Abbreviation_map().keys():
                            diag = PhysioNetDataset.SNOMEDCT_To_Abbreviation_map()[diag]
                            info[diag] = 1
                elif l[0] in extra_fields:
                    info[l[0][1:]] = l[-1]
        
            if len(sr) > 1:
                raise RuntimeError(f"Found inconsistend gain per lead in {file_id}")

            info.update(baseline_correction)
            info['Gain'] = float(sr.pop().split("/")[0]) * 1e-6

        except Exception as e:
            print(file_content)
            raise e

        return info

    @staticmethod
    def convert_sex(sex):
        if sex in ('Female', 'female', 'F', 'f'):
            return 0
        elif sex in ('Male', 'male', 'M', 'm'):
            return 1
        return float('nan')

    def convert_to_numpy(self, df, path):
        for filename in tqdm(df['Filename']):
            file_path = PhysioNetDataset.split_to_path(filename, steps=3)
            if not os.path.exists(os.path.join(path, file_path, f'{filename}.npy')):
                data = io.loadmat(os.path.join(path, file_path, f'{filename}.mat',))['val']
                np.save(os.path.join(path, file_path, f'{filename}.npy'), data)

    def extract_archive(
        self,
        remove_finished: bool = False,
    ):
        if self.to_path is None:
            self.to_path = os.path.dirname(self.from_path)

        print("Now extracting data from tar archive, this may take a while...")

        process = subprocess.Popen(f'tar zxf {self.from_path} -C {self.to_path}', shell=True)
        process.wait()

        print(f"Done extracting - exit code = {process.returncode}")

        print("Now extracting header info and moving files")
        header_info = []

        for file in os.listdir(f'{self.to_path}/{self.archive_root}'):
            full_file_path = f'{self.to_path}/{self.archive_root}/{file}'
            file_id, ftype = file.split('.')

            # Only read header files, extract raw data in header file is found
            if ftype == 'hea':
                with open(full_file_path, 'r') as extracted:
                    # Get header info from header file
                    header_info.append(self.extract_header_info(extracted.readlines(), file_id))

                mat_file_name = f'{self.to_path}/{self.archive_root}/{file_id}.mat'

                # Create folder structure for raw data
                file_path = PhysioNetDataset.split_to_path(file_id, steps=3)
                f_to_path = os.path.join(self.to_path, file_path)

                # Move files
                os.renames(mat_file_name, os.path.join(f_to_path, f'{file_id}.mat'))
                os.renames(full_file_path, os.path.join(f_to_path, f'{file_id}.hea'))

        header_df = pd.DataFrame(header_info)
        header_df.to_csv(f"{self.to_path}/header_info.csv")

        print("All done")

        if remove_finished:
            os.remove(self.from_path)


class PTBXLDataset(PhysioNetDataset):
    def __init__(self, path : str = "./PTB_XL", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_PTBXL.tar.gz', 
                path, 
                filename='PTB_XL.tar.gz',
                md5='55e8a5c25eadfeff4fcd38f5bbf3cb13'
            )

        self.archive_root = 'WFDB_PTBXL'
        self.from_path = f'{path}/PTB_XL.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(PTBXLDataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )


class CPSC2018Dataset(PhysioNetDataset):
    def __init__(self, path : str = "./CPSC_2018", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_CPSC2018.tar.gz', 
                path, 
                filename='CPSC_2018.tar.gz',
                md5='5d4b520e3b6558a33dc9dbe49d08f8f1'
            )

        self.archive_root = 'WFDB_CPSC2018'
        self.from_path = f'{path}/CPSC_2018.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(CPSC2018Dataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )


class China12LeadDataset(PhysioNetDataset):
    def __init__(self, path : str = "./China_12_Lead", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_CPSC2018_2.tar.gz', 
                path, 
                filename='China_12_Lead.tar.gz',
                md5='5b1498abacaa1b5a762691c006e737ad'
            )

        self.archive_root = 'WFDB_CPSC2018_2'
        self.from_path = f'{path}/China_12_Lead.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(China12LeadDataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )


class INCARTDataset(PhysioNetDataset):
    def __init__(self, path : str = "./INCART", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_StPetersburg.tar.gz', 
                path, 
                filename='INCART.tar.gz',
                md5='525dde6bd26bff0dcb35189e78ae7d6d'
            )

        self.archive_root = 'WFDB_StPetersburg'
        self.from_path = f'{path}/INCART.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(INCARTDataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )


class PTBDataset(PhysioNetDataset):
    def __init__(self, path : str = "./PTB", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_PTB.tar.gz', 
                path, 
                filename='PTB.tar.gz',
                md5='3df4662a8a9189a6a5924424b0fcde0e'
            )

        self.archive_root = 'WFDB_PTB'
        self.from_path = f'{path}/PTB.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(PTBDataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )


class Georgia12LeadDataset(PhysioNetDataset):
    def __init__(self, path : str = "./Georgia_12_Lead", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_Ga.tar.gz', 
                path, 
                filename='Georgia_12_Lead.tar.gz',
                md5='47085dd62baca5ace4041025d6910b13'
            )

        self.archive_root = 'WFDB_Ga'
        self.from_path = f'{path}/Georgia_12_Lead.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(Georgia12LeadDataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )


class ChapmanUniversityDataset(PhysioNetDataset):
    def __init__(self, path : str = "./Chapman_University", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_ChapmanShaoxing.tar.gz', 
                path, 
                filename='Chapman_University.tar.gz',
                md5='a3e10171eba1e7520a38919594d834e5'
            )

        self.archive_root = 'WFDB_ChapmanShaoxing'
        self.from_path = f'{path}/Chapman_University.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(ChapmanUniversityDataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )


class NingboDataset(PhysioNetDataset):
    def __init__(self, path : str = "./Ningbo", download : bool = True, use_numpy: bool = True, *args, **kwargs):
        self.download = download
        self.use_numpy = use_numpy

        if download:
            download_url(
                'https://storage.googleapis.com/physionetchallenge2021-public-datasets/WFDB_Ningbo.tar.gz', 
                path, 
                filename='Ningbo.tar.gz',
                md5='84171145922078146875394acb89b765'
            )

        self.archive_root = 'WFDB_Ningbo'
        self.from_path = f'{path}/Ningbo.tar.gz'
        self.to_path = None

        # Assume extraction was done correctly if header_info csv exists
        # TODO, is this enough? 
        if not os.path.exists(f"{path}/header_info.csv"):
            self.extract_archive()

        df = pd.read_csv(f"{path}/header_info.csv")

        if use_numpy:
            self.convert_to_numpy(df, path)

        super(NingboDataset, self).__init__(
            dataset_function='physionet_numpy' if use_numpy else 'physionet',
            waveform_dir=path,
            dataset=df,
            *args,
            **kwargs
        )