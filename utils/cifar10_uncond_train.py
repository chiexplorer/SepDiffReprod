import json, os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm

from utils.tools import load_model, get_epoch_from_path
from models.ddpm import DDPM

"""
    CIFAR10数据集，DDPM模型 无条件模式 训练
"""

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hpath = r"D:\Projects\pyprog\SepDiffReprod\configs\config.json"
    with open(hpath, 'r') as f:
        hparams = json.load(f)

    # 数据加载
    dataset = CIFAR10(
        root='../save/CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, **hparams["dataloader_opts"], shuffle=True, drop_last=True, pin_memory=True)

    # 模型
    model = DDPM(hparams['unet_config'])  # , conditioning_key=hparams['conditioning_key']
    ckpt = hparams['ckpt_path'] if 'ckpt_path' in hparams else None
    epoch_start = 0  # 起始epoch
    # checkpoint加载
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        state_dict = load_model(ckpt)
        epoch_start = get_epoch_from_path(ckpt) + 1
        model.load_state_dict(state_dict)
    model.cuda()

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams["learning_rate"], weight_decay=1e-4)
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=hparams["epoch"], eta_min=0, last_epoch=-1)

    # logger
    writer = SummaryWriter('../logs')

    # 训练
    best_loss = float('inf')
    print(f"------从epoch [{epoch_start}]开始继续训练------")
    for e in range(epoch_start, hparams["epoch"]):
        epoch_loss = 0.0
        num_batches = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                #train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss, _ = model(x_0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams["grad_clip"])  # 梯度裁剪
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                # 记录loss均值
                epoch_loss += loss.item()
                num_batches += 1
        epoch_loss /= num_batches
        writer.add_scalar('loss', epoch_loss, global_step=e)
        # save checkpoints
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(
                hparams["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))
            print(f"最佳loss更新了, Epoch [{e}], new best loss [{epoch_loss}]")
