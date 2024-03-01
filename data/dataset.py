import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchaudio
import pandas as pd
from tqdm import tqdm
from utils.speech import get_mel


class DmTestTrain(Dataset):
    def __init__(self, csv_file, config=None):
        self.data = []
        self.config = config
        self.df = pd.read_csv(csv_file)
        self.resampler = torchaudio.transforms.Resample(orig_freq=self.config['orig_sample_rate'],
                                                           new_freq=self.config['sampling_rate'])
        self.get_mel()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_mel(self):
        for i, record in tqdm(self.df.iterrows()):
            item = {}
            try:
                mix, sr = torchaudio.load(record['mix_wav'])
                s1, _ = torchaudio.load(record['s1_wav'])
                s2, _ = torchaudio.load(record['s2_wav'])
                mix = self.resampler(mix)
                s1 = self.resampler(s1)
                s2 = self.resampler(s2)
                mix_mel = get_mel(mix, self.config)
                s1_mel = get_mel(mix, self.config)
                s2_mel = get_mel(mix, self.config)
                item['mix_wav'] = mix
                item['s1_wav'] = s1
                item['s2_wav'] = s2
                item['mix_mel'] = mix_mel
                item['s1_mel'] = s1_mel
                item['s2_mel'] = s2_mel
                self.data.append(item)
            except Exception as e:
                print(f"record {i + 1} Exception occured: { e }")


if __name__ == '__main__':
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    fpath = r"D:\Projects\pyprog\SepDiffReprod\save\aishell1mix2_test.csv"
    hparams = {
        'orig_sample_rate': 16000,
        'sampling_rate': 22050,
        'n_fft': 1024,
        "hop_size": 256,
        "win_size": 1024,
        "num_mels": 80,
        "fmin": 0,
        "fmax": 8000,
        "dataloader_opts": {
            "batch_size": 1,
            "num_workers": 4
        }
    }
    # hparams = AttrDict(hparams)
    dataset = DmTestTrain(fpath, hparams)
    print("data count: ", len(dataset))
    print("first element: ", dataset[0])
    loader = DataLoader(dataset, shuffle=False,
                        sampler=None, pin_memory=True,
                        drop_last=True, **hparams['dataloader_opts'])
