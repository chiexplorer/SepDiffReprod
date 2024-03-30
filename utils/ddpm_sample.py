import json, os
import numpy as np
from omegaconf import OmegaConf

from models.ddpm import DDPM
from ldm.models.diffusion.ddpm import DDPMCustom
from models.ddim import DDIMSampler
import torch
from torchvision.utils import save_image
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

def load_model(ckpt, k=None):
    if ckpt and k is None:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
    elif ckpt and k is not None:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        pl_sd = pl_sd[k]
    else:
        pl_sd = {"state_dict": None}

    return pl_sd


if __name__ == '__main__':
    mode = "custom"  # light 为基于lightning训练; custom 为自己搭建的流程训练
    print(f"-----run with mode [{mode}]-----")
    if mode == "custom":
        ckptpath = r"D:\Projects\pyprog\SepDiffReprod\save\ckpt_2_.pt"
        hpath = r"D:\Projects\pyprog\SepDiffReprod\configs\config.json"
        img_savepath = r"D:\Projects\pyprog\SepDiffReprod\save\imgs\sampledImg.png"
        npz_savepath = r"D:\Projects\pyprog\SepDiffReprod\save\samples.npz"

        with open(hpath, 'r') as f:
            hparams = json.load(f)
        # 载入预训练参数
        state_dict = load_model(ckptpath)
        # 加载模型
        model = DDPM(hparams['unet_config'])  # , conditioning_key=hparams['conditioning_key']
        model.load_state_dict(state_dict, strict=False)
        model.cuda()
        model.eval()

        # 采样——DDIM sampler
        sampler = DDIMSampler(model)
        sample_opt = hparams["sample_opt"]
        shape = [sample_opt["channels"], sample_opt["image_size"], sample_opt["image_size"]]
        samples, intermediates = sampler.sample(sample_opt['timesteps'], batch_size=sample_opt['batch_size'],
                                                shape=shape, eta=1.0, verbose=False,)
        # 存为图片
        for i, x in enumerate(samples):
            img = custom_to_pil(x)
            imgpath = f"D:\Projects\pyprog\SepDiffReprod\save\imgs\sampled_{i}.png"
            img.save(imgpath)
        # imgs = custom_to_np(samples)
        # samples = samples.detach().cpu()
        # samples = torch.clamp(samples, -1, 1)
        # samples = samples * 0.5 + 0.5
        # save_image(samples, img_savepath, nrow=4)
        # np.savez(npz_savepath, sample=imgs)

        # # 采样
        # sample_opt = hparams["sample_opt"]
        # imgs = model.sample(sample_opt["batch_size"])
        # save_image(imgs, savepath, nrow=4)
    elif mode == "light":
        ckptpath = r"D:\Projects\pyprog\SepDiffReprod\logs\2024-03-08T21-20-22_cifar10\checkpoints\epoch=019.ckpt"
        hpath = r"D:\Projects\pyprog\SepDiffReprod\configs\cifar10.yaml"

        hparams = OmegaConf.load(hpath)
        hparams = hparams["model"]["params"]
        # 载入预训练参数
        state_dict = load_model(ckptpath, 'state_dict')
        # 加载模型
        model = DDPMCustom(hparams['unet_config'], linear_start=hparams['linear_start'],
                           linear_end=hparams['linear_end'], timesteps=1000,
                           first_stage_key=hparams['first_stage_key'], image_size=hparams['image_size'],
                           channels=hparams['channels'], monitor=hparams['monitor'], ckpt_path=ckptpath)
        model.load_state_dict(state_dict, strict=False)
        model.cuda()
        model.eval()

        # 采样——DDIM sampler
        sampler = DDIMSampler(model)
        sample_opt = {
            "timesteps": 1000,
            "batch_size": 4,
            "channels": 3,
            "image_size": 32
        }
        shape = [sample_opt["channels"], sample_opt["image_size"], sample_opt["image_size"]]
        samples, intermediates = sampler.sample(sample_opt['timesteps'], batch_size=sample_opt['batch_size'],
                                                shape=shape, eta=1.0, verbose=False, )
        # 存为图片
        for i, x in enumerate(samples):
            img = custom_to_pil(x)
            imgpath = f"D:\Projects\pyprog\SepDiffReprod\save\imgs\sampled_{i}.png"
            img.save(imgpath)