# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from enum import Enum
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from omegaconf.listconfig import ListConfig
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate_yang(x, shift, scale):
    return x * (1 + scale) + shift

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

# class DiTBlock(nn.Module):
#     """
#     A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )
#
#     def forward(self, x, c):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
#         x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
#         x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads=-1, dim_head=-1, mlp_ratio=4.0, dropout=0., context_dim=None, **block_kwargs):
        super().__init__()
        if num_heads == -1:  # 检查多头attn的必须参数
            assert dim_head != -1, 'Either num_heads or dim_head has to be set'
        if dim_head == -1:
            assert num_heads != -1, 'Either num_heads or dim_head has to be set'
        # 自动确定多头attn的参数
        if dim_head == -1:
            dim_head = hidden_size // num_heads
        else:
            num_heads = hidden_size // dim_head  # 注意力头数
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.attn = SpatialTransformer(in_channels=hidden_size, n_heads=num_heads, d_head=dim_head, **block_kwargs)  # 使用cross-attn替代原attn
        self.attn = BasicTransformerBlock(dim=hidden_size, n_heads=num_heads, d_head=dim_head,
                                          dropout=dropout, context_dim=context_dim)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.attn(self.norm1(x))  # ln, self-attn
        x = x + self.attn(self.norm2(x), c)  # ln, cross-attn
        x = x + self.mlp(self.norm3(x))
        return x

class DiTBlockE2E(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads=-1, dim_head=-1, mlp_ratio=4.0, dropout=0., context_dim=None, **block_kwargs):
        super().__init__()
        if num_heads == -1:  # 检查多头attn的必须参数
            assert dim_head != -1, 'Either num_heads or dim_head has to be set'
        if dim_head == -1:
            assert num_heads != -1, 'Either num_heads or dim_head has to be set'
        # 自动确定多头attn的参数
        if dim_head == -1:
            dim_head = hidden_size // num_heads
        else:
            num_heads = hidden_size // dim_head  # 注意力头数
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.attn = SpatialTransformer(in_channels=hidden_size, n_heads=num_heads, d_head=dim_head, **block_kwargs)  # 使用cross-attn替代原attn
        self.attn = BasicTransformerBlock(dim=hidden_size, n_heads=num_heads, d_head=dim_head,
                                          dropout=dropout, context_dim=context_dim)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.attn(self.norm1(x))  # ln, self-attn
        x = x + self.attn(self.norm2(x), c)  # ln, cross-attn
        x = x + self.mlp(self.norm3(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if isinstance(patch_size, int):
            out_dim = patch_size * patch_size * out_channels
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list) or isinstance(patch_size, ListConfig):
            out_dim = patch_size[0] * patch_size[1] * out_channels
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # 若c是三维形式
        if len(c.shape) == 3:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)  # 在最后一个维度拆分
            # # 在embed_dim复制一份
            # shift = torch.cat([shift, shift], dim=1)
            # scale = torch.cat([scale, scale], dim=1)
            x = modulate_yang(self.norm_final(x), shift, scale)  # 均值&方差和x同shape
        elif len(c.shape) == 2:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)  # 原代码
            x = modulate(self.norm_final(x), shift, scale)
        else:
            raise ValueError(f"Num of dim(dim={len(c.shape)}) about c is not supported.")

        x = self.linear(x)
        return x

class FinalLayerE2E(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # 若c是二维形式
        if len(c.shape) == 3:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)  # 在最后一个维度拆分
            x = modulate_yang(self.norm_final(x), shift, scale)  # 均值&方差和x同shape
        elif len(c.shape) == 2:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)  # 原代码
            x = modulate(self.norm_final(x), shift, scale)
        else:
            raise ValueError(f"Num of dim(dim={len(c.shape)}) about c is not supported.")

        x = self.linear(x)
        return x


class DiT(nn.Module):
    """ Diffusion model with a Transformer backbone. """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        custom_shape=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.custom_shape = custom_shape  # 扩展到宽高比不固定的情形

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if custom_shape:
            # 编码mix mel spec以在W维度拼接时可用
            self.y_embedder = PatchEmbed((input_size[0] / 2, input_size[1]),
                                         patch_size, in_channels, hidden_size, bias=True)
        else:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)  # 不需要类编码器了

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # patch位置编码, shape (1, grid[0]*grid[1], hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.y_pos_embed = nn.Parameter(torch.zeros(1, num_patches//2, hidden_size), requires_grad=False)  # y的位置编码

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # 初始化所有线性层，令：weight—Xavier 均匀分布，bias—全零值
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.custom_shape:  # 扩展的情形
            pos_embed = get_2d_sincos_pos_embed_yang(self.pos_embed.shape[-1],
                                                     int(self.input_size[0] / self.patch_size[0]),
                                                     int(self.input_size[1] / self.patch_size[1]))
            y_pos_embed = get_2d_sincos_pos_embed_yang(self.pos_embed.shape[-1],
                                                       int(self.input_size[0] / self.patch_size[0] / 2),
                                                       int(self.input_size[1] / self.patch_size[1]))
            self.y_pos_embed.data.copy_(torch.from_numpy(y_pos_embed).float().unsqueeze(0))
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        if self.custom_shape:
            # # 扩展代码
            p_h = self.patch_size[0]
            p_w = self.patch_size[1]
            h = int(self.input_size[0] / p_h)
            w = int(self.input_size[1] / p_w)
            assert h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, c))  # (B, h, w, p, p, c)
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p_h, w * p_w))
        else:
            # 原代码
            p = self.x_embedder.patch_size[0]
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))  # (B, h, w, p, p, c)
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))

        return imgs

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        context: (N,) tensor of context condition
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        t = torch.unsqueeze(t, dim=1)  # 临时 保证t和y可加，但未思考其意义
        if self.custom_shape:
            context = self.y_embedder(context) + self.y_pos_embed   # (N, T_c, D)
        else:
            context = self.y_embedder(context, self.training)    # (N, D)

        c = t + context  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        if self.custom_shape:
            # 临时 将c在dim=1处复制一份，保证FinalLayer的adaLN 可用
            c = torch.cat((c, c), dim=1)
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, context, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, context)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class DiTE2E(nn.Module):
    """ Diffusion model with a Transformer backbone. """
    def __init__(
        self,
        input_size=257,
        in_channels=4,
        hidden_size=1152,
        emb_dim=256,
        depth_t=4,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        custom_shape=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size  # 输入长度
        self.emb_dim = emb_dim  # 输出编码长度
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.custom_shape = custom_shape  # 扩展到宽高比不固定的情形
        self.x_embedder = nn.Linear(input_size, hidden_size, bias=True)
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.x_embedder = nn.Conv1d(in_channels, hidden_size, kernel_size=1, stride=1, padding=0)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)  # 不需要类编码器了
        # 临时，编码mix mel spec以在W维度拼接时可用
        # self.y_embedder = PatchEmbed((input_size[0] / 2, input_size[1]),
        #                              patch_size, in_channels, hidden_size, bias=True)
        # self.y_embedder = nn.Conv1d(in_channels//2, hidden_size//2, kernel_size=1, stride=1, padding=0)
        self.y_embedder = nn.Linear(input_size, hidden_size, bias=True)
        # Will use fixed sin-cos embedding:
        # token位置编码, shape (1, in_channels, input_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_channels, hidden_size), requires_grad=False)
        self.y_pos_embed = nn.Parameter(torch.zeros(1, in_channels//2, hidden_size), requires_grad=False)  # y的位置编码
        self.blocks_t = nn.ModuleList([
            DiTBlockE2E(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_t)
        ])  # 时间步融合crossAttn
        self.blocks = nn.ModuleList([
            DiTBlockE2E(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])  # 条件融合crossAttn
        # self.final_layer = FinalLayerE2E(hidden_size, input_size)  // 原final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.SiLU(),
            nn.Linear(hidden_size, input_size, bias=True)
        )
        # self.initialize_weights()
        # self.init_pos_embed()  # 用不上位置编码

    def init_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.pos_embed.shape[-2], dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # 初始化所有线性层，令：weight—Xavier 均匀分布，bias—全零值
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.pos_embed.shape[-2], dtype=np.float32))
        y_pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.pos_embed.shape[-2]//2, dtype=np.float32))
        self.y_pos_embed.data.copy_(torch.from_numpy(y_pos_embed).float().unsqueeze(0))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        context: (N,) tensor of context condition
        """
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        t = torch.unsqueeze(t, dim=1)  # 临时 保证t和y可加，但未思考其意义
        # context = self.y_embedder(context, self.training)    # (N, D)
        context = self.y_embedder(context)   # (N, T_c, D)
        # 融合时间步信息
        for block in self.blocks_t:
            x = block(x, t)  # (N, T, D)
        # 融合条件信息
        for block in self.blocks:
            x = block(x, context)  # (N, T, D)
        x = self.final_layer(x)  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward_with_cfg(self, x, t, context, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, context)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class DiTE2EV2(nn.Module):
    """ Diffusion model with a Transformer backbone. """
    def __init__(
        self,
        input_size=32,
        patch_size=1,
        in_channels=4,
        hidden_size=1152,
        depth_t=4,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbedE2E(input_size, 1, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 编码mix mel spec以在W维度拼接时可用
        self.y_embedder = PatchEmbedE2E(input_size,
                                     1, in_channels, hidden_size, bias=True)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # patch位置编码, shape (1, grid[0]*grid[1], hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches//2, hidden_size), requires_grad=False)
        self.y_pos_embed = nn.Parameter(torch.zeros(1, num_patches//2, hidden_size), requires_grad=False)  # y的位置编码

        self.blocks_t = nn.ModuleList([
            DiTBlockE2E(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_t)
        ])  # 时间步融合crossAttn
        self.blocks = nn.ModuleList([
            DiTBlockE2E(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # self.initialize_weights()  # 初始化位置编码 + zero init model params
        self.init_pos_embed()  # 初始化位置编码

    # # 在通道方向拼接源
    # def init_pos_embed(self):
    #     pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, np.arange(self.input_size, dtype=np.float32))
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # 在编码方向拼接源
    def init_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, np.arange(self.input_size // 2, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # 初始化所有线性层，令：weight—Xavier 均匀分布，bias—全零值
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, np.arange(self.input_size, dtype=np.float32))
        # y_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size,
        #                                               np.arange(self.input_size, dtype=np.float32))
        # self.y_pos_embed.data.copy_(torch.from_numpy(y_pos_embed).float().unsqueeze(0))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        context: (N,) tensor of context condition
        """
        x = self.x_embedder(x) + torch.cat([self.pos_embed, self.pos_embed], dim=1)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        c = self.y_embedder(context) + self.pos_embed   # (N, T, D)
        t = torch.unsqueeze(t, dim=1)  # 临时 保证t和y可加，但未思考其意义

        # c = t + c  # (N, D)
        # 融合时间步信息
        for block in self.blocks_t:
            x = block(x, t)  # (N, T, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        # if self.custom_shape:
        #     # 临时 将c在dim=1处复制一份，保证FinalLayer的adaLN 可用
        #     c = torch.cat((c, c), dim=1)
        x = self.final_layer(x, c)               # (N, T, C)
        x = torch.transpose(x, 1, 2)  # (N, T, C) -> (N, C, T)
        return x

    def forward_with_cfg(self, x, t, context, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, context)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class PatchEmbedE2E(nn.Module):
    """
        2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=1,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=False,
            output_fmt=None,
            bias=True,
            strict_img_size=True,
            dynamic_img_pad=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        if img_size is not None:
            self.img_size = img_size
        else:
            self.img_size = None
        self.num_patches = img_size  # patch数量即为输入长度
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=1, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert(L == self.img_size, f"Input length ({L}) doesn't match model ({self.img_size}).")
            elif not self.dynamic_img_pad:
                assert(
                    L % self.patch_size == 0,
                    f"Input length ({L}) should be divisible by patch size ({self.patch_size})."
                )

        x = self.proj(x)  # in_channel -> emb_dim
        x = x.transpose(1, 2)  # NCL -> NLC
        # if self.output_fmt != Format.NCHW:
        #     x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # 生成坐标的grid
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_yang(embed_dim, w, h, cls_token=False, extra_tokens=0):
    """
    grid_w: int of the grid width
    grid_h: int of the grid height
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # 生成坐标的grid
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, w, h])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """ 根据坐标集grid和码字字长embed_dim生成位置编码 """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    # 长，宽各用一半的码字
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    # 创建sin | cos的周期序列，长度为码字的一半(D/2)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    # 计算正余弦编码
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    # 拼接正余弦编码
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


if __name__ == '__main__':
    from torchinfo import summary
    import netron
    import torch.onnx
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # # PatchEmbed测试
    # x = torch.randn(2, 1, 10, 20, dtype=torch.float)
    # x = x.to(device)
    # model = PatchEmbed(img_size=(10, 20), patch_size=(5, 5), in_chans=1, embed_dim=768)
    # model.cuda()
    # pred = model(x)
    # print(pred.shape)

    # # # 2d-position-embed——扩展到长宽不一致测试
    # emb_dim = 768
    # grid_w = 2
    # grid_h = 4
    # pos_embedding = get_2d_sincos_pos_embed_yang(emb_dim, grid_w, grid_h)
    # print(pos_embedding.shape)

    # # # 1d-position-embed测试
    # grid = np.arange(300, dtype=np.float32)
    # pos_emb_dim = 768
    # pos_emb = get_1d_sincos_pos_embed_from_grid(pos_emb_dim, grid)
    # print("encodes shape: ", pos_emb.shape)

    # # #  DiTBlock测试
    # block = DiTBlock(hidden_size=768,
    #     num_heads=16,
    #     mlp_ratio=4.0
    # )
    # block.cuda()
    # x = torch.randn(3, 256, 768).to(device)
    # c = torch.randn(3, 1, 768).to(device)
    # h = block(x, c)
    # print(h.shape)

    # # # DIT测试——原始配置
    # x = torch.randn(2, 3, 224, 224)
    # t = torch.randint(0, 1000, (2, )).to(device)
    # y = torch.randint(0, 1000, (2, )).to(device)
    # x = x.to(device)
    # model = DiT(input_size=224, patch_size=16, in_channels=3, hidden_size=1024, depth=12, num_heads=12)
    # model.cuda()
    # outputs = model(x, t, y)
    # pred, _ = torch.split(outputs, x.shape[1], dim=1)
    # print(pred.shape)

    # # # DIT测试——模仿UNet的输入
    # x = torch.randn(2, 256, 10, 20)
    # t = torch.randint(0, 1000, (2, )).to(device)
    # y = torch.randint(0, 1000, (2, )).to(device)
    # c = torch.randn(2, 256, 5, 20).to(device)  # mel 条件
    # x = x.to(device)
    # model = DiT(input_size=(10, 20), patch_size=(5, 5), in_channels=256, hidden_size=768, depth=12, num_heads=12, custom_shape=True)
    # model.cuda()
    # outputs = model(x, t, c)
    # pred = outputs
    # print(pred.shape)

    # # # DIT测试——E2E测试
    # input_size = 300
    # x = torch.randn(2, 16, input_size).to(device)
    # t = torch.randint(0, 1000, (2, )).to(device)
    # y = torch.randn(2, 8, input_size).to(device)
    # # c = torch.randn(2, 256, 5, 20).to(device)  # mel 条件
    # model = DiTE2E(input_size=input_size, in_channels=16, hidden_size=768, depth=12, num_heads=12, custom_shape=False)
    # # summary(model, input_size=((2, 16, 257),(2, ), (2, 8, 257)))  # , mode="train"
    # model.cuda()
    # outputs = model(x, t, y)
    # pred = outputs
    # print(pred.shape)

    # # DIT测试——E2E测试
    input_size = 300
    x = torch.randn(2, 16, input_size).to(device)
    t = torch.randint(0, 1000, (2, )).to(device)
    y = torch.randn(2, 8, input_size).to(device)
    # c = torch.randn(2, 256, 5, 20).to(device)  # mel 条件
    model = DiTE2EV2(input_size=input_size, in_channels=16, hidden_size=768, depth_t=2, depth=12, num_heads=12)
    # summary(model, input_size=((2, 16, 257),(2, ), (2, 8, 257)))  # , mode="train"
    model.cuda()
    outputs = model(x, t, y)
    pred = outputs
    print(pred.shape)

    # netron可视化
    # modelData = "./DiTE2E.pth"
    # torch.onnx.export(model, (x, t, y), modelData)
    # netron.start(modelData)
