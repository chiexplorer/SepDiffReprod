import os, tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from encodec.model import EncodecModel
from ldm.data.librimix import LibriMixE2E

"""
    检查E2E encodec方法中，定位条件音频信息损失发生的阶段
"""

def norm_min_max(x, minV, maxV):
    """
        将x值域通过min-max标准化到值域[-1, 1]内
    """
    mean = (maxV + minV) / 2
    x = x - mean  # 减去区间均值
    x = x / mean  # 归一化
    # x = (x - minV) / (maxV - minV)  # min-max标准化
    # torch.clamp(x, min=0, max=1)  # 钳位
    # x = 2 * x - 1  # 值域缩放到[-1, 1]
    return x

def norm_min_max_inv(x, minV, maxV):
    """
        将x值域从[-1, 1]通过逆标准化到值域[minV, maxV]内
    """
    mean = (maxV + minV) / 2
    x = x * mean  # 归一化
    x = x + mean  # 减去区间均值
    # x = torch.clamp(x, -1, 1)  # [-1, 1]
    # x = (x + 1) / 2  # [0, 1]
    # x = x * (maxV - minV) + minV  # 逆min-max标准化
    return x


if __name__ == '__main__':
    dirpath = r"H:\exp\dataset\LibriMixE2E\Libri2Mix\wav16k\min\debug\mix_clean"
    filelist = os.listdir(dirpath)
    savedir = r"D:\Projects\pyprog\SepDiffReprod\test\files"
    save_sr = 16000
    fixed_len = 96000

    # 载入encodec
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    # 工具准备
    resampler = torchaudio.transforms.Resample(orig_freq=model.sample_rate, new_freq=save_sr)
    loss_fn = torch.nn.MSELoss()  # MSE func
    # ## 检查——prepare输出
    # for fname in filelist:
    #     fpath = os.path.join(dirpath, fname)
    #     savepath = os.path.join(savedir, fname.replace(".npy", ".wav"))
    #     wav_np = np.load(fpath)
    #     print("np shape: ", wav_np.shape)
    #     wav_np = torch.from_numpy(wav_np)
    #     wav_np = [(wav_np.to(torch.int64), None), ]
    #
    #     with torch.no_grad():
    #         wav = model.decode(wav_np)
    #     wav = wav[..., :fixed_len]
    #     wav = resampler(wav)
    #     wav = torch.squeeze(wav, dim=0)
    #     print("wav recon: ", wav.shape)
    #     torchaudio.save(savepath, wav, save_sr)

    # # 检查——loader之后
    hparams_e2e = {
        'csv_path': "D:/Projects/pyprog/SepDiffReprod/data/encodec/libri2mix_debug.csv",
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
            "batch_size": 1,
            "num_workers": 0
        }
    }
    savedir = os.path.join(savedir, "loader")
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    dataset = LibriMixE2E(hparams_e2e)
    print("data count: ", len(dataset))
    loader = DataLoader(dataset, shuffle=False,
                        sampler=None, pin_memory=True,
                        drop_last=True, **hparams_e2e['dataloader_opts'])
    for data in loader:
        savepath = os.path.join(savedir, data['fname'][0]+".wav")
        mix = data['mix_wav']
        mix_wav = norm_min_max_inv(mix, 0, 1023)
        mix_wav = [(mix_wav.to(torch.int64), None), ]

        with torch.no_grad():
            wav = model.decode(mix_wav)
        wav = wav[..., :fixed_len]
        wav = resampler(wav)
        wav = torch.squeeze(wav, dim=0)
        print("wav recon: ", wav.shape)
        torchaudio.save(savepath, wav, save_sr)