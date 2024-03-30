import json, os
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from utils.speech import align_audio, get_mel
import logging

"""
    从Librimix数据集提取mel spec并转存到同结构的目录，处理过程包括下列约束：
        1. 保证帧数为320帧，通过对齐音频长度
        2. 保证采样率为22050（受BigVGAN Vocoder限制）
        3. 存储文件格式为.npy
"""

if __name__ == '__main__':
    dirpath = r"G:\Resource\Datasets\LibriMix\Libri2Mix\wav16k\min\train-100"  # 训练集
    dest_dirpath = r"D:\Data\LibriMixAligned\Libri2Mix\wav16k\min\train-100"  # 训练集
    # dirpath = r"G:\Resource\Datasets\LibriMix\Libri2Mix\wav16k\min\test"  # 测试集
    # dest_dirpath = r"D:\Data\LibriMixAligned\Libri2Mix\wav16k\min\test"  # 测试集
    # dirpath = r"G:\Resource\Datasets\LibriMix\Libri2Mix\wav16k\min\dev"  # 验证集
    # dest_dirpath = r"D:\Data\LibriMixAligned\Libri2Mix\wav16k\min\dev"  # 验证集

    hpath = r"D:\Projects\pyprog\SepDiffReprod\configs\LibriMixConfig.json"
    logpath = "./logs/prepare_librimix_mel.log"
    opt_dir = ['mix_clean', 's1', 's2']  # 需要处理的文件夹
    with open(hpath, 'r') as f:
        hparams = json.load(f)

    # 准备工具
    logging.basicConfig(filename=logpath, level=logging.INFO)
    target_len = hparams['fixed_len']  # 固定长度
    # 重采样器
    resampler = torchaudio.transforms.Resample(orig_freq=hparams['orig_sample_rate'],
                                            new_freq=hparams['sampling_rate'])

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
        mel_max_values = []
        for aname in tqdm(audio_list):
            try:
                apath = os.path.join(subdir_orig, aname)  # 音频路径
                # 读音频，取mel spec，存为npy文件
                mix, _ = torchaudio.load(apath)
                # 1. 重采样
                mix = resampler(mix)
                # 2. 对齐音频长度
                mix = align_audio(mix, target_len)
                # 3. 取mel spec
                mix_mel, mix_mel_max = get_mel(mix, hparams)
                mel_max_values.append(mix_mel_max)
                # 4. 存为.npy
                savename = os.path.splitext(aname)[0] + '.npy'
                savepath = os.path.join(subdir_dest, savename)
                # np.save(savepath, mix_mel.detach().numpy())  # 临时注释
            except Exception as e:
                print(f"处理 [{aname}] 时出错了", e)
                logging.info(f"dir: {subdir}, audio: {aname}")
        mel_max_values = np.array(mel_max_values)
        np.save(f"dev_{subdir}_mel_spec_max.npy", mel_max_values)
        print("子集{}--处理完毕!".format(subdir))
        break

    # # # 测试 npy文件
    # # 取 npy文件
    # np_mix_mel = np.load("mix_mel.npy")
    # print("np_mix_mel shape: ", np_mix_mel.shape)
    #
    # # 绘制npy mel spec图
    # plt.imshow(np_mix_mel[0])
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()


