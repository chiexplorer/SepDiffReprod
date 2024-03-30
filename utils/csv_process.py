import csv
import pandas as pd
import torchaudio
import sys
sys.path.append("../")

from utils.speech import get_mel

hparams = {
    'orig_sample_rate': 16000,
    'sampling_rate': 22050,
    'n_fft': 1024,
    "hop_size": 256,
    "win_size": 1024,
    "num_mels": 80,
    "fmin": 0,
    "fmax": 8000,
}

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == '__main__':
    fpath = r"../save/aishell1mix2_test.csv"
    # f = open(fpath, 'r')
    # data = csv.reader(f)
    # for i,row in enumerate(data):
    #     print(row)
    #     if i > 5:
    #         break
    #
    # f.close()

    df = pd.read_csv(fpath)
    count = 0
    it = df.iterrows()
    dataset = []
    hparams = AttrDict(hparams)
    for i, record in it:
        item = {}
        print(record['mix_wav'], record['s1_wav'], record['s2_wav'], record['noise_wav'])
        resampler = torchaudio.transforms.Resample(orig_freq=hparams['orig_sample_rate'],
                                                   new_freq=hparams['sampling_rate'])
        mix, sr = torchaudio.load(record['mix_wav'])
        s1, _ = torchaudio.load(record['s1_wav'])
        s2, _ = torchaudio.load(record['s2_wav'])
        mix = resampler(mix)
        s1 = resampler(s1)
        s2 = resampler(s2)
        mix_mel = get_mel(mix, hparams)
        s1_mel = get_mel(mix, hparams)
        s2_mel = get_mel(mix, hparams)
        item['mix_wav'] = mix
        item['s1_wav'] = s1
        item['s2_wav'] = s2
        item['mix_mel'] = mix_mel
        item['s1_mel'] = s1_mel
        item['s2_mel'] = s2_mel
        dataset.append(item)
        count += 1
        if count > 2:
            break