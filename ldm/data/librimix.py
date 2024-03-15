import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchaudio
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

    def init_data(self):
        for i, record in tqdm(self.df.iterrows()):
            item = {}
            # load from mel file
            if self.use_mel_file:
                try:
                    mix_mel = np.load(record['mix_wav'])
                    s1_mel = np.load(record['s1_wav'])
                    s2_mel = np.load(record['s1_wav'])
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
                    # mix_mel = get_mel(mix, self.config)
                    # s1_mel = get_mel(s1, self.config)
                    # s2_mel = get_mel(s2, self.config)
                    # Mel spec时，用不到wav
                    item['mix_wav'] = mix
                    item['s1_wav'] = s1
                    item['s2_wav'] = s2

                    # # E2E时，用不到mel spec
                    # item['mix_mel'] = mix_mel
                    # item['s1_mel'] = s1_mel
                    # item['s2_mel'] = s2_mel
                    self.data.append(item)
                except Exception as e:
                    print(f"record ID {record['ID']} Exception occured: { e }")


if __name__ == '__main__':
    # # audio file
    fpath = r"D:\Projects\pyprog\SepDiffReprod\data\wav\libri2mix_train-100-test.csv"
    # # npy file
    # fpath = r"D:\Projects\pyprog\SepDiffReprod\save\libri2mix_test.csv"
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

    dataset = LibriMixData(hparams)
    print("data count: ", len(dataset))
    print("first element: ", dataset[0])
    loader = DataLoader(dataset, shuffle=False,
                        sampler=None, pin_memory=True,
                        drop_last=True, **hparams['dataloader_opts'])