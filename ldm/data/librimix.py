import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchaudio
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.speech import get_mel, align_audio

"""
    Librimix Dataset, 
"""


class LibriMixData(Dataset):
    """
        @desc: Dataset for mel spec, generated according to config. Support load from wav or npy files.
        @note: Audio will automaticly resample from [orig_sample_rate] to [sampling_rate]].
    """
    def __init__(self, config):
        self.data = []
        self.config = config
        self.use_mel_file = config['use_mel_file']  # 是否使用mel npy文件
        if self.use_mel_file:
            self.mel_min = config['mel_min']
            self.mel_max = config['mel_max']
        self.df = pd.read_csv(config['mel_path']) if self.use_mel_file else pd.read_csv(config['csv_path'])
        self.fixed_len = config['fixed_len'] if "fixed_len" in config else None
        self.image_size = config['image_size'] if "image_size" in config else None
        self.resampler = None if config['orig_sample_rate'] == config['sampling_rate'] \
            else torchaudio.transforms.Resample(orig_freq=self.config['orig_sample_rate'],
                                                           new_freq=self.config['sampling_rate'])
        self.init_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess_custom_image(self, x):
        """
            将频谱标准化到值域[-1, 1]内
            @params x: numpy ndarray，(log) mel spec
            @params mel_min: min value of log mel spec
            @params mel_max: max value of log mel spec
        """
        x = (x - self.mel_min) / (self.mel_max - self.mel_min)  # min-max标准化
        torch.clamp(x, min=0, max=1)  # 钳位
        x = 2 * x - 1  # 值域缩放到[-1, 1]
        return x

    def init_data(self):
        for i, record in tqdm(self.df.iterrows()):
            item = {}
            # load from mel file
            if self.use_mel_file:
                try:
                    mix_mel = np.load(record['mix_wav'])
                    s1_mel = np.load(record['s1_wav'])
                    s2_mel = np.load(record['s2_wav'])
                    mix_mel = self.preprocess_custom_image(torch.from_numpy(mix_mel))
                    s1_mel = self.preprocess_custom_image(torch.from_numpy(s1_mel))
                    s2_mel = self.preprocess_custom_image(torch.from_numpy(s2_mel))
                    item['mix_mel'] = mix_mel
                    item['s1_mel'] = s1_mel
                    item['s2_mel'] = s2_mel
                    self.data.append(item)
                except Exception as e:
                    print(f"record ID {record['ID']} Exception occured: { e }")
            # load from wav file
            else:
                try:
                    # read audios
                    mix, sr = torchaudio.load(record['mix_wav'])
                    s1, _ = torchaudio.load(record['s1_wav'])
                    s2, _ = torchaudio.load(record['s2_wav'])
                    # resample, if necessary
                    if self.resampler is not None:
                        mix = self.resampler(mix)
                        s1 = self.resampler(s1)
                        s2 = self.resampler(s2)
                    # align sample length, if necessary
                    if self.fixed_len is not None:
                        mix = align_audio(mix, self.fixed_len)
                        s1 = align_audio(s1, self.fixed_len)
                        s2 = align_audio(s2, self.fixed_len)
                    # reshape, if necessary
                    if self.image_size is not None:
                        mix = mix.view(1, self.image_size, self.image_size)
                        s1 = s1.view(1, self.image_size, self.image_size)
                        s2 = s2.view(1, self.image_size, self.image_size)
                    # # E2E时，用不到mel
                    # mix_mel = get_mel(mix, self.config)
                    # s1_mel = get_mel(s1, self.config)
                    # s2_mel = get_mel(s2, self.config)
                    # Mel spec时，用不到wav
                    item['mix_wav'] = mix
                    item['s1_wav'] = s1
                    item['s2_wav'] = s2
                    item['path'] = os.path.splitext(os.path.basename(record['mix_wav']))[0]
                    # # E2E时，用不到mel spec
                    # item['mix_mel'] = mix_mel
                    # item['s1_mel'] = s1_mel
                    # item['s2_mel'] = s2_mel
                    self.data.append(item)
                except Exception as e:
                    print(f"record ID {record['ID']} Exception occured: { e }")

class LibriMixTest(Dataset):
    """
        @desc: Dataset for mel spec test
        @note: Audio will automaticly resample from [orig_sample_rate] to [sampling_rate]].
    """
    def __init__(self, config):
        self.data = []
        self.config = config
        self.mel_min = config['mel_min']
        self.mel_max = config['mel_max']
        self.use_mel_file = config['use_mel_file']  # 是否使用mel npy文件
        self.df = pd.read_csv(config['mel_path']) if self.use_mel_file else pd.read_csv(config['csv_path'])
        self.fixed_len = config['fixed_len'] if "fixed_len" in config else None
        self.image_size = config['image_size'] if "image_size" in config else None
        self.resampler = None if config['orig_sample_rate'] == config['sampling_rate'] \
            else torchaudio.transforms.Resample(orig_freq=self.config['orig_sample_rate'],
                                                           new_freq=self.config['sampling_rate'])
        self.init_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess_custom_image(self, x):
        """
            将频谱标准化到值域[-1, 1]内
            @params x: numpy ndarray，(log) mel spec
            @params mel_min: min value of log mel spec
            @params mel_max: max value of log mel spec
        """
        x = (x - self.mel_min) / (self.mel_max - self.mel_min)  # min-max标准化
        torch.clamp(x, min=0, max=1)  # 钳位
        x = 2 * x - 1  # 值域缩放到[-1, 1]
        return x

    def init_data(self):
        for i, record in tqdm(self.df.iterrows()):
            item = {}
            # load from mel file
            if self.use_mel_file:
                try:
                    mix_mel = np.load(record['mix_wav'])
                    s1_mel = np.load(record['s1_wav'])
                    s2_mel = np.load(record['s2_wav'])
                    mix_mel = self.preprocess_custom_image(torch.from_numpy(mix_mel))
                    s1_mel = self.preprocess_custom_image(torch.from_numpy(s1_mel))
                    s2_mel = self.preprocess_custom_image(torch.from_numpy(s2_mel))
                    item['mix_mel'] = mix_mel
                    item['s1_mel'] = s1_mel
                    item['s2_mel'] = s2_mel
                    item['path'] = os.path.splitext(os.path.basename(record['mix_wav']))[0]
                    self.data.append(item)
                except Exception as e:
                    print(f"record ID {record['ID']} Exception occured: { e }")
            # load from wav file
            else:
                try:
                    # read audios
                    mix, sr = torchaudio.load(record['mix_wav'])
                    s1, _ = torchaudio.load(record['s1_wav'])
                    s2, _ = torchaudio.load(record['s2_wav'])
                    # resample, if need
                    if self.resampler is not None:
                        mix = self.resampler(mix)
                        s1 = self.resampler(s1)
                        s2 = self.resampler(s2)
                    # align sample length, if need
                    if self.fixed_len is not None:
                        mix = align_audio(mix, self.fixed_len)
                        s1 = align_audio(s1, self.fixed_len)
                        s2 = align_audio(s2, self.fixed_len)
                    # reshape
                    if self.image_size is not None:
                        mix = mix.view(1, self.image_size, self.image_size)
                        s1 = s1.view(1, self.image_size, self.image_size)
                        s2 = s2.view(1, self.image_size, self.image_size)
                    mix_mel = get_mel(mix, self.config)
                    s1_mel = get_mel(s1, self.config)
                    s2_mel = get_mel(s2, self.config)
                    # Mel spec时，用不到wav
                    # item['mix_wav'] = mix
                    item['s1_wav'] = s1
                    item['s2_wav'] = s2

                    # # E2E时，用不到mel spec
                    item['mix_mel'] = mix_mel
                    item['s1_mel'] = s1_mel
                    item['s2_mel'] = s2_mel
                    self.data.append(item)
                except Exception as e:
                    print(f"record ID {record['ID']} Exception occured: { e }")

class LibriMixE2E(Dataset):
    """
        @desc: Dataset for Encodec output, generated according to config. Loading from npy files.
        @note: Assuming that npy files were extracted from sr=24000, length=96000 wav.
            More details refers to utils/prepare_librimix_e2e.py.
    """
    def __init__(self, config):
        self.data = []
        self.config = config
        self.norm_min = config['norm_min']
        self.norm_max = config['norm_max']
        self.df = pd.read_csv(config['csv_path'])  # csv with npy file pairs record in.
        self.fixed_len = config['fixed_len'] if "fixed_len" in config else None  # fixed length of audio
        # self.resampler = None if config['orig_sample_rate'] == config['sampling_rate'] \
        #         #     else torchaudio.transforms.Resample(orig_freq=self.config['orig_sample_rate'],
        #         #                                                    new_freq=self.config['sampling_rate'])
        self.init_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess_encodec_output(self, x):
        """
            样本标准化
        """
        # --- min-max norm -> [0, 1]
        x = (x - self.norm_min) / (self.norm_max - self.norm_min)
        # # --- /mu norm -> [-1, 1]
        # mean = (self.norm_max + self.norm_min) / 2
        # x = x - mean  # 减去区间均值
        # x = x / mean  # 归一化
        # x = (x - self.norm_min) / (self.norm_max - self.norm_min)  # min-max标准化
        # torch.clamp(x, min=0, max=1)  # 钳位
        # x = 2 * x - 1  # 值域缩放到[-1, 1]
        return x

    def init_data(self):
        for i, record in tqdm(self.df.iterrows()):
            item = {}
            # load from mel file
            try:
                mix_mel = np.load(record['mix_wav'])
                s1_mel = np.load(record['s1_wav'])
                s2_mel = np.load(record['s2_wav'])
                # 规范化->[-1, 1]
                mix_mel = self.preprocess_encodec_output(torch.from_numpy(mix_mel))
                s1_mel = self.preprocess_encodec_output(torch.from_numpy(s1_mel))
                s2_mel = self.preprocess_encodec_output(torch.from_numpy(s2_mel))
                item['mix_wav'] = torch.squeeze(mix_mel).to(torch.float32)
                item['s1_wav'] = torch.squeeze(s1_mel).to(torch.float32)
                item['s2_wav'] = torch.squeeze(s2_mel).to(torch.float32)
                item['fname'] = os.path.splitext(os.path.basename(record['mix_wav']))[0]
                self.data.append(item)
            except Exception as e:
                print(f"record ID {record['ID']} Exception occured: { e }")


if __name__ == '__main__':
    # # audio file
    # fpath = r"D:\Projects\pyprog\SepDiffReprod\data\wav\libri2mix_train-100-test.csv"
    # # npy file
    # fpath = r"D:\Projects\pyprog\SepDiffReprod\save\libri2mix_test.csv"
    # # encodec file
    fpath = r"D:\Projects\pyprog\SepDiffReprod\data\encodec\libri2mix_debug.csv"
    use_mel = False
    hparams = {
        'csv_path': fpath,
        'mel_path': fpath,
        "use_mel_file": use_mel,
        'orig_sample_rate': 16000,
        'sampling_rate': 16000,
        'n_fft': 1024,
        "hop_size": 256,
        "win_size": 1024,
        "num_mels": 80,
        "fmin": 0,
        "fmax": 8000,
        "fixed_len": 65536,
        "image_size": 256,
        "dataloader_opts": {
            "batch_size": 1,
            "num_workers": 0
        }
    }

    hparams_e2e = {
        'csv_path': fpath,
        'orig_sample_rate': 16000,
        'sampling_rate': 16000,
        'n_fft': 1024,
        "hop_size": 256,
        "win_size": 1024,
        "num_mels": 80,
        "fmin": 0,
        "fmax": 8000,
        "fixed_len": 65536,
        "image_size": 256,
        'norm_min': 0,
        'norm_max': 1023,
        "dataloader_opts": {
            "batch_size": 2,
            "num_workers": 0
        }
    }

    dataset = LibriMixE2E(hparams_e2e)
    print("data count: ", len(dataset))
    print("first element: ", dataset[0])
    loader = DataLoader(dataset, shuffle=False,
                        sampler=None, pin_memory=True,
                        drop_last=True, **hparams_e2e['dataloader_opts'])