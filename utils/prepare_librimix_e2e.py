import json, os
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import logging

from utils.speech import align_audio, get_mel
from encodec import EncodecModel
from encodec.utils import save_audio, convert_audio

"""
    从Librimix数据集提取encodec 编码并转存到同结构的目录，处理过程包括下列约束：
        1. duration为4s
        2. 保证采样率为24000（受Encodec限制）
        3. 存储文件格式为.npy
"""

if __name__ == '__main__':
    mode = "process"  # "process"为处理音频；"test"为测试npy文件

    # dirpath = r"G:\Resource\Datasets\LibriMix\Libri2Mix\wav16k\min\train-100"  # 训练集
    # dest_dirpath = r"D:\Data\LibriMixAligned\Libri2Mix\wav16k\min\train-100"  # 训练集
    # dirpath = r"G:\Resource\Datasets\LibriMix\Libri2Mix\wav16k\min\test"  # 测试集
    # dest_dirpath = r"D:\Data\LibriMixAligned\Libri2Mix\wav16k\min\test"  # 测试集
    # dirpath = r"G:\Resource\Datasets\LibriMix\Libri2Mix\wav16k\min\dev"  # 验证集
    # dest_dirpath = r"D:\Data\LibriMixAligned\Libri2Mix\wav16k\min\dev"  # 验证集
    # dirpath = r"H:\exp\dataset\LibriMix\Libri2Mix\wav16k\min\train-100"  # debug集
    # dest_dirpath = r"H:\exp\dataset\LibriMixE2E\Libri2Mix\wav16k\min\train-100"  # debug集
    dirpath = r"H:\exp\dataset\LibriMix\Libri2Mix\wav16k\min\test"  # 测试集
    dest_dirpath = r"H:\exp\dataset\LibriMixE2E\Libri2Mix\wav16k\min\test"  # 测试集

    # 载入配置参数
    hpath = r"D:\Projects\pyprog\SepDiffReprod\configs\LibriMixE2E.json"
    opt_dir = ['mix_clean', 's1', 's2']  # 需要处理的文件夹
    with open(hpath, 'r') as f:
        hparams = json.load(f)
    # 检查输出目录
    if not os.path.exists(dest_dirpath):
        os.mkdir(dest_dirpath)
    if mode == "process":
        # 准备工具
        target_len = hparams['fixed_len']  # 固定长度
        resampler = torchaudio.transforms.Resample(orig_freq=hparams['orig_sample_rate'],
                                                new_freq=hparams['sampling_rate'])  # 重采样器
        model = EncodecModel.encodec_model_24khz()  # 编码器
        model.set_target_bandwidth(6.0)

        # 目录操作
        subdirs = os.listdir(dirpath)
        for subdir in subdirs:
            if subdir not in opt_dir:
                print(f"跳过对 [{subdir}] 子集的处理.")
                continue
            # 创建子目录
            subdir_orig = os.path.join(dirpath, subdir)
            subdir_dest = os.path.join(dest_dirpath, subdir)
            os.makedirs(subdir_dest, exist_ok=True)

            # 处理子目录的音频
            audio_list = os.listdir(subdir_orig)
            print("开始处理子集--{}".format(subdir))
            for aname in tqdm(audio_list):
                try:
                    apath = os.path.join(subdir_orig, aname)  # 音频路径
                    # 读音频，取mel spec，存为npy文件
                    mix, _ = torchaudio.load(apath)
                    # 1. 重采样
                    mix = resampler(mix)
                    # 2. 对齐音频长度
                    mix = align_audio(mix, target_len)
                    mix = torch.unsqueeze(mix, dim=0)
                    # 3. 取encodec编码
                    with torch.no_grad():
                        encoded_frames = model.encode(mix)
                    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
                    codes = codes.detach().numpy()  # 转numpy
                    codes = codes.astype(np.int16)  # 优化存储
                    # 4. 存为.npy
                    savename = os.path.splitext(aname)[0] + '.npy'
                    savepath = os.path.join(subdir_dest, savename)
                    np.save(savepath, codes)  # 临时注释
                except Exception as e:
                    print(f"处理 [{aname}] 时出错了", e)
                    logging.info(f"dir: {subdir}, audio: {aname}")
            print("子集{}--处理完毕!".format(subdir))
    elif mode == "test":
        # # 测试 npy文件
        # 取 npy文件
        fpath = r"H:\exp\dataset\LibriMixE2E\Libri2Mix\wav16k\min\debug\s2\84-121123-0005_3536-23268-0024.npy"
        savedir = "./"
        savepath = os.path.join(savedir,
                        os.path.splitext(os.path.basename(fpath))[0] + "_recon.wav")
        np_mix_codes = np.load(fpath)
        print("np_mix_mel shape: ", np_mix_codes.shape)

        resampler = torchaudio.transforms.Resample(orig_freq=hparams['sampling_rate'],
                                                new_freq=hparams['orig_sample_rate'])  # 重采样器
        fixed_len = hparams['fixed_len']
        model = EncodecModel.encodec_model_24khz()  # 编码器
        model.set_target_bandwidth(6.0)

        try:
            # 绘制npy mel spec图
            plt.imshow(np_mix_codes[0])
            # plt.colorbar()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Plot npy Failed: ", e)

        # 解码回音频
        codes = torch.from_numpy(np_mix_codes).to(torch.int64)
        frame = [(codes, None), ]
        with torch.no_grad():
            wav_recon = model.decode(frame)
        # 重采样回原采样率
        wav_recon = resampler(wav_recon)
        save_audio(wav_recon[0, ...], savepath, hparams['orig_sample_rate'], rescale=False)
    else:
        print("-----Nothing to do.-----")

