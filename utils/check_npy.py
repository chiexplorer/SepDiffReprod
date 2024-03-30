import os, json, logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from types import SimpleNamespace
from utils.speech import get_mel

"""
    检查mel转存的npy文件值域，可绘制为频谱图
"""

MEL_MIN = -11.5129
MEL_MAX = 1.0908
def custom_to_pil(x):
    """
    Convert torch tensor to a PIL
    """
    x = (x - MEL_MIN) / (MEL_MAX - MEL_MIN)
    print(f"未钳位的mel spec min: {x.min()}, max: {x.max()}")
    x = np.clip(x, 0., 1.)  # 钳位保证值域正确性
    x = np.transpose(x, (1, 2, 0))
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x[:, :, 0])
    # if not x.mode == "RGB":
    #     x = x.convert("RGB")
    return x


if __name__ == '__main__':
    # 取 npy文件, 存为log mel spec图
    np_mix_mel = np.load("../test/mix_mel.npy")
    print("np_mix_mel shape: ", np_mix_mel.shape)
    print("np_mix_mel:", np_mix_mel.max(), np_mix_mel.min())
    # # 绘制npy mel spec图
    # plt.imshow(np_mix_mel[0])
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    img = custom_to_pil(np_mix_mel)
    print("img: ", img)
    img.save("mix_mel.png")

    # # 分析最大mel spec取值的分布
    # fpath = r"D:\Projects\pyprog\SepDiffReprod\test\dev_mix_clean_mel_spec_max.npy"  # mix_clean
    # # fpath = r"D:\Projects\pyprog\SepDiffReprod\test\dev_s1_mel_spec_max.npy"  # s1
    # data = np.load(fpath)
    #
    # # 使用 Matplotlib 绘制直方图
    # plt.hist(data, bins=30, edgecolor='black')  # 设置直方图的边界颜色为黑色
    # plt.title('Distribution of Data')  # 设置标题
    # plt.xlabel('Value')  # 设置 x 轴标签
    # plt.ylabel('Frequency')  # 设置 y 轴标签
    # plt.grid(True)  # 添加网格线
    # plt.show()  # 显示图形
