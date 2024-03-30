import os, json, logging
import torch
import torch.nn as nn
import numpy as np


"""
    验证norm和逆向norm是否导致信息损失
"""

def loader_preprocess(x, minV, maxV):
    mean = (maxV + minV) / 2
    x = x - mean  # 减去区间均值
    x = x / mean  # 归一化
    # x = (x - minV) / (maxV - minV)  # min-max标准化
    # torch.clamp(x, min=0, max=1)  # 钳位
    # x = 2 * x - 1  # 值域缩放到[-1, 1]
    # x = 2 * x
    return x


# def norm_min_max(x, minV, maxV):
#     """
#         将x值域通过min-max标准化到值域[-1, 1]内
#     """
#     x = (x - minV) / (maxV - minV)  # min-max标准化
#     torch.clamp(x, min=0, max=1)  # 钳位
#     x = 2 * x - 1  # 值域缩放到[-1, 1]
#     return x
#
# def norm_min_max_inv(x, minV, maxV):
#     """
#         将x值域从[-1, 1]通过逆标准化到值域[minV, maxV]内
#     """
#     mean = (maxV + minV) / 2
#     x = x * mean  # 归一化
#     x = x + mean  # 减去区间均值
#     # x = torch.clamp(x, -1, 1)  # [-1, 1]
#     # x = (x + 1) / 2  # [0, 1]
#     # x = x / 2
#     # x = x * (maxV - minV) + minV  # 逆min-max标准化
#     return x

def norm_min_max(x, minV, maxV):
    x = (x - minV) / (maxV - minV)
    return x

def norm_min_max_inv(x, minV, maxV):
    x = x * (maxV - minV) + minV  # 逆min-max标准化
    return x


if __name__ == '__main__':

    # 模拟输入
    x = torch.arange(0, 1024).to(torch.int16)
    # x = torch.randint(0, 1024, (1, 1, 80000)).to(torch.int16)

    # 模拟norm
    x_norm = norm_min_max(x,0, 1023)

    # 模拟逆向norm
    x_recon = norm_min_max_inv(x_norm, 0, 1023)
    x_recon = x_recon.to(torch.int16)

    print("是否相等: ", x.equal(x_recon))
    not_equal = torch.ne(x, x_recon)
    unequal_elements = x[not_equal]
    print("不相等的元素&个数: ", unequal_elements, torch.numel(unequal_elements))

    # # 计算MSE
    # loss_fn = torch.nn.MSELoss()
    # mse_loss = loss_fn(x, x_recon)
    # print("MSE loss: ", mse_loss)
    # if mse_loss < 1e-6:
    #     print("无信息损失")
    # else:
    #     print("有信息损失")
