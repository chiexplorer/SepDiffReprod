import os
import numpy as np
import torch
from PIL import Image


# LDM的采样输出转np array函数
def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def custom_to_pil(x):
    """
    Convert torch tensor to a PIL
    """
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)  # 钳位保证值域正确性
    x = (x + 1.) / 2.  # 值域 [0, 1]
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def load_model(ckpt):
    """
        Load model from ckpt file path
        :param ckpt:
        :return: state_dict, params of trained model
    """
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}
    state_dict = pl_sd

    return state_dict

def get_epoch_from_path(ckpt):
    """
        Get epoch from ckpt file path
        :example: ckpt = "D:\Projects\pyprog\SepDiffReprod\save\ckpt_19_.pt", return 19
        :return: epoch, int
    """
    dirname, filename = os.path.split(ckpt)

    num_str = ''
    # 遍历字符串，将所有数字字符拼接到 num_str 中
    for char in filename:
        if char.isdigit():
            num_str += char
    return int(num_str)