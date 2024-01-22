from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import ConvModule, DropPath
from mmcv.runner import BaseModule, Sequential

from ..builder import BACKBONES
from ..utils import InvertedResidual, SELayer, make_divisible
import clip


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

@BACKBONES.register_module()
class VisionTransformer(BaseModule):
    def __init__(self, 
                 frozen_stages=0,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(
                         type='Constant',
                         layer=['_BatchNorm', 'GroupNorm'],
                         val=1)
                 ]):
        super(VisionTransformer, self).__init__(init_cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/16", device=device)

        self.frozen_stages = frozen_stages
        self.norm_eval = False
        
        self.input_resolution = model.visual.input_resolution
        self.output_dim = model.visual.output_dim
        self.conv1 = model.visual.conv1

        scale = 768 ** -0.5
        self.class_embedding = nn.Parameter(model.visual.class_embedding)
        self.positional_embedding = nn.Parameter(model.visual.positional_embedding)
        self.ln_pre = LayerNorm(768)

        self.transformer = model.visual.transformer

        self.ln_post = LayerNorm(768)
        self.proj = nn.Parameter(model.visual.proj)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x.type(torch.float16))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return tuple([x, x])
    
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            pass
            # m = self.layers[i]
            # m.eval()
            # for param in m.parameters():
            #     param.requires_grad = False

    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()