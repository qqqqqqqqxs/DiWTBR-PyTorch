import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from einops import rearrange
from torch.autograd import Function
import pywt
# from model.torch_wavelets import DWT_2D, IDWT_2D

# class DWT_Function(Function):
#     @staticmethod
#     def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
#         x = x.contiguous().to(dtype=torch.float16)
#         ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
#         ctx.shape = x.shape

#         dim = x.shape[1]
#         x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
#         x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
#         x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
#         x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
#         x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1).to(dtype=torch.float32)
#         return x

#     @staticmethod
#     def backward(ctx, dx):
#         if ctx.needs_input_grad[0]:
#             w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
#             B, C, H, W = ctx.shape
#             dx = dx.view(B, 4, -1, H//2, W//2).to(dtype=torch.float16)

#             dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
#             filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
#             dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C).to(dtype=torch.float32)

#         return dx, None, None, None, None

# class IDWT_Function(Function):
#     @staticmethod
#     def forward(ctx, x, filters):
#         ctx.save_for_backward(filters)
#         ctx.shape = x.shape

#         B, _, H, W = x.shape
#         x = x.view(B, 4, -1, H, W).transpose(1, 2).to(dtype=torch.float16)
#         C = x.shape[1]
#         x = x.reshape(B, -1, H, W)
#         filters = filters.repeat(C, 1, 1, 1)
#         x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
#         return x.to(dtype=torch.float32)

#     @staticmethod
#     def backward(ctx, dx):
#         if ctx.needs_input_grad[0]:
#             filters = ctx.saved_tensors
#             filters = filters[0]
#             B, C, H, W = ctx.shape
#             C = C // 4
#             dx = dx.contiguous().to(dtype=torch.float16)

#             w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
#             x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
#             x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
#             x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
#             x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
#             dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1).to(dtype=torch.float32)
#         return dx, None

# class IDWT_2D(nn.Module):
#     def __init__(self, wave):
#         super(IDWT_2D, self).__init__()
#         w = pywt.Wavelet(wave)
#         rec_hi = torch.Tensor(w.rec_hi)
#         rec_lo = torch.Tensor(w.rec_lo)
        
#         w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
#         w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
#         w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
#         w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

#         w_ll = w_ll.unsqueeze(0).unsqueeze(1)
#         w_lh = w_lh.unsqueeze(0).unsqueeze(1)
#         w_hl = w_hl.unsqueeze(0).unsqueeze(1)
#         w_hh = w_hh.unsqueeze(0).unsqueeze(1)
#         filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
#         self.register_buffer('filters', filters)
#         self.filters = self.filters.to(dtype=torch.float16)

#     def forward(self, x):
#         return IDWT_Function.apply(x, self.filters)

# class DWT_2D(nn.Module):
#     def __init__(self, wave):
#         super(DWT_2D, self).__init__()
#         w = pywt.Wavelet(wave)
#         dec_hi = torch.Tensor(w.dec_hi[::-1]) 
#         dec_lo = torch.Tensor(w.dec_lo[::-1])

#         w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
#         w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
#         w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
#         w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

#         self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

#         self.w_ll = self.w_ll.to(dtype=torch.float16)
#         self.w_lh = self.w_lh.to(dtype=torch.float16)
#         self.w_hl = self.w_hl.to(dtype=torch.float16)
#         self.w_hh = self.w_hh.to(dtype=torch.float16)

#     def forward(self, x):
#         return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


###float32
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()  
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)  
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)  

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)  
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x  

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()  

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)  
    
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

def _get_relative_position_index(height: int, width: int) -> torch.Tensor: 
    coords = torch.stack(torch.meshgrid([torch.arange(height), torch.arange(width)]))
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        squeeze_ratio: float,
        stride: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        p_stochastic_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        proj: Sequence[nn.Module]
        self.proj: nn.Module

        should_proj = stride != 1 or in_channels != out_channels 
        if should_proj:
            proj = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj  
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()  

        mid_channels = int(out_channels * expansion_ratio) 
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")  
        else:
            self.stochastic_depth = nn.Identity()  
        _layers = OrderedDict()
        _layers["pre_norm"] = norm_layer(in_channels)
        _layers["conv_a"] = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=activation_layer,
            # norm_layer= None,
            norm_layer=norm_layer,
            inplace=None,
        )
        _layers["conv_b"] = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation_layer=activation_layer,
            # norm_layer= None,
            norm_layer=norm_layer,
            groups=mid_channels,
            inplace=None,
        )
        _layers["squeeze_excitation"] = SqueezeExcitation(mid_channels, sqz_channels, activation=nn.SiLU) 
        _layers["conv_c"] = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True)

        self.layers = nn.Sequential(_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H / stride, W / stride].
        """
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x

class WaveAttention(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        max_seq_len: int,
        n: float,
        patch_h: int,
        patch_w:int,
        partition_type: str,
        focusing_factor=3, 
        kernel_size=5,
        Flatten = False,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")
        self.n_heads = feat_dim // head_dim #2 4 8 16
        self.patch_h = patch_h
        self.patch_w = patch_w
        if Flatten == True:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((self.n_heads, 1, 1))))
        self.head_dim = head_dim #32
        self.max_seq_len = max_seq_len #256
        self.focusing_factor = focusing_factor #3
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//4, kernel_size=1, padding=0, stride=1),
            #nn.BatchNorm2d(feat_dim//4),
            #nn.GELU()
            #nn.ReLU(inplace=True),#nn.GELU()
        )
        self.filter = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, stride=1, groups=1),
            #nn.BatchNorm2d(feat_dim),
            #nn.GELU()
            #nn.ReLU(inplace=True),#nn.GELU()
        )
        # self.dwc = nn.Conv2d(in_channels=head_dim//4, out_channels=head_dim//4, kernel_size=kernel_size,
        #                      groups=head_dim//4, padding=kernel_size // 2)
        self.q = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.kv = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2)
        )
        self.proj = nn.Linear(feat_dim+feat_dim//4, feat_dim)
        self.linear1 = nn.Linear(head_dim+head_dim//4, head_dim)
        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3) #64 2*32*3
        self.scale_factor = feat_dim**-0.5
        self.flatten = Flatten
        self.n = n
        self.partition_type = partition_type
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, self.n_heads, bias=False)
        )
        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(((self.patch_h*2 - 1) * (self.patch_w*2 - 1), self.n_heads), dtype=torch.float32),
        )

        self.register_buffer("relative_position_index", _get_relative_position_index(self.patch_h, self.patch_w))
        # initialize with truncated normal the bias
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor: 
        bias_index = self.relative_position_index.view(-1)  # 64 64
        #print(bias_index.shape)#64*64
        relative_bias = self.relative_position_bias_table[bias_index].view(self.max_seq_len, self.max_seq_len, -1)  # type: ignore
        #print(relative_bias.shape)#64 64 8
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        #print(relative_bias.shape)#8 64 64
        return relative_bias.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, G, P, D].
        Returns:
            Tensor: Output tensor with expected layout of [B, G, P, D].
        """
        B, G, P, D = x.shape 
        H, DH = self.n_heads, self.head_dim# 2 4 8 16/32
        SP = P // 4
        if self.partition_type == "window":
            w = h = int(P ** 0.5)
        else:
            w = int((P/self.n) ** 0.5)
            h = int((P/self.n) ** 0.5*self.n)
        if self.flatten == False:
            q = self.q(x)
            x = x.view(B, G, P, D).permute(0, 1, 3, 2)# B D G P
            x = rearrange(x, "B G D (w h) -> (B G) D w h", w=w, h=h)
            x_dwt = self.dwt(self.reduce(x))
            x_dwt = self.filter(x_dwt)
            x_idwt = self.idwt(x_dwt)
            x_idwt = rearrange(x_idwt, "(B G) D w h -> B G D (w h)", B=B, G=G)
            x_dwt = rearrange(x_dwt, "(B G) D w_2 h_2 -> B G D (w_2 h_2)", B=B, G=G)
            x_idwt = x_idwt.view(B, G, -1, P).permute(0, 1, 3, 2)#B G P D
            x_dwt = self.norm(x_dwt.view(B, G, D, SP).permute(0, 1, 3, 2))

            kv = self.kv(x_dwt)
            k, v = torch.chunk(kv, 2, dim=-1)

            q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
            k = k.reshape(B, G, SP, H, DH).permute(0, 1, 3, 2, 4)
            v = v.reshape(B, G, SP, H, DH).permute(0, 1, 3, 2, 4)
            k = k * self.scale_factor
            dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k) 
            dot_prod = F.softmax(dot_prod, dim=-1)
            out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
            pos_bias = self.get_relative_positional_bias()
            out_bias = torch.einsum("X H I J, B G H J D -> B G H I D", pos_bias, out)
            out = out + out_bias
            out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)
            out = self.proj(torch.cat([out, x_idwt], dim=-1))
        out = self.merge(out)#64 49 64
        return out

class SwapAxes(nn.Module):
    """Permute the axes of a tensor."""

    def __init__(self, a: int, b: int) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.swapaxes(x, self.a, self.b)
        return res


class WindowPartition(nn.Module):
    """
    Partition the input tensor into non-overlapping windows.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int, p1: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
            p (int): Number of partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, H/P, W/P, P*P, C].
        """
        B, C, H, W = x.shape
        P = p  #7
        P1 = p1 
        # chunk up H and W dimensions
        x = x.reshape(B, C, H // P, P, W // P1, P1) #p=16
        x = x.permute(0, 2, 4, 3, 5, 1)
        # colapse P * P dimension
        x = x.reshape(B, (H // P) * (W // P1), P * P1, C)
        return x 


class WindowDepartition(nn.Module):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int, h_partitions: int, w_partitions: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        B, G, PP, C = x.shape
        P = p 
        HP, WP = h_partitions, w_partitions
        # split P * P dimension into 2 P tile dimensionsa
        x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        x = x.permute(0, 5, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)# 64 7*8 7*8
        return x

class DilatedPartition(nn.Module):
    """
    Partition the input tensor into non-overlapping windows.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int, p1: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
            p (int): Number of partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, H/P, W/P, P*P, C].
        """
        B, C, H, W = x.shape
        P = p #7
        P1 = p1
        # chunk up H and W dimensions
        x = x.reshape(B, C, H // P, P, W // P1, P1) #p=16
        x = x.permute(0, 2, 4, 3, 5, 1)
        # colapse P * P dimension
        x = x.reshape(B, (H // P) * (W // P1), P * P1, C)
        return x 
    
class DilatedDepartition(nn.Module):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int, h_partitions: int, w_partitions: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        # split P * P dimension into 2 P tile dimensionsa
        x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        x = x.permute(0, 5, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)# 64 7*8 7*8
        return x
    
class PartitionAttentionLayer(nn.Module):
    """
    Layer for partitioning the input tensor into non-overlapping windows and applying attention to each window.

    Args:
        in_channels (int): Number of input channels.
        head_dim (int): Dimension of each attention head.
        partition_size (int): Size of the partitions.
        partition_type (str): Type of partitioning to use. Can be either "dilation" or "window".
        dilation_size (Tuple[int, int]): Size of the dilation to partition the input tensor into.
        mlp_ratio (int): Ratio of the  feature size expansion in the MLP layer.
        activation_layer (Callable[..., nn.Module]): Activation function to use.
        norm_layer (Callable[..., nn.Module]): Normalization function to use.
        attention_dropout (float): Dropout probability for the attention layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        p_stochastic_dropout (float): Probability of dropping out a partition.
    """

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        # partitioning parameters
        partition_size: int, 
        partition_type: str,
        # dilation size needs to be known at initialization time
        # because we need to know hamy relative offsets there are in the dilation
        dilation_size: Tuple[int, int],
        mlp_ratio: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        attention_dropout: float,
        mlp_dropout: float,
        p_stochastic_dropout: float,
        Flatten = False
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = dilation_size[0] // partition_size 
        self.n_partitions1 = dilation_size[1] // partition_size
        self.partition_type = partition_type
        self.dilation_size = dilation_size
        self.n = dilation_size[1]/dilation_size[0]
        self.partition_size = partition_size 
        self.Flatten = Flatten

        if partition_type not in ["dilation", "window"]:
            raise ValueError("partition_type must be either 'dilation' or 'window'")

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions 
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.window_partition = WindowPartition()
        self.dilation_partition = DilatedPartition()
        self.window_departition = WindowDepartition()
        self.dilation_departition = DilatedDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "dilation" else nn.Identity() 
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "dilation" else nn.Identity()
        if partition_type == "window":
            self.attn_layer = nn.Sequential(
                norm_layer(in_channels),
                WaveAttention(in_channels, 
                                                     head_dim, 
                                                     partition_size**2,
                                                     n=self.n,
                                                     partition_type = partition_type,
                                                     patch_h = self.partition_size,
                                                     patch_w = self.partition_size,
                                                     Flatten = self.Flatten
                                                     ),
                
                nn.Dropout(attention_dropout),
            )
        else:
            self.attn_layer = nn.Sequential(
                norm_layer(in_channels),
                WaveAttention(in_channels, 
                                                     head_dim, 
                                                     self.n_partitions1*self.n_partitions,
                                                     n=self.n,
                                                     partition_type = partition_type,
                                                     patch_h = self.n_partitions,
                                                     patch_w = self.n_partitions1,
                                                     Flatten = self.Flatten
                                                     ),
                
                nn.Dropout(attention_dropout),
            )

        # pre-normalization similar to transformer layers
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout),
        )

        # layer scale factors
        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")

    def forward(self, x: Tensor) -> Tensor:
        gh, gw = self.n_partitions, self.n_partitions1
        torch._assert(
            self.dilation_size[0] % self.partition_size == 0 and self.dilation_size[1] % self.partition_size == 0,
            "dilation size must be divisible by partition size. Got dilation size of {} and partition size of {}".format(
                self.dilation_size, self.partition_size
            ),
        )
        #print(self.n)
        if self.partition_type == "window":
            x = self.window_partition(x, self.partition_size, self.partition_size) 
        else:
            x = self.dilation_partition(x, self.partition_size, self.partition_size)
        x = self.partition_swap(x)# dilation：B 256 1024 64
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))
        x = self.departition_swap(x)
        if self.partition_type == "window":
            x = self.window_departition(x, self.partition_size, gh, gw) 
        else:
            x = self.dilation_departition(x, self.partition_size, gh, gw)

        return x


class DiWTLayer(nn.Module):
    def __init__(
        self,
        # conv parameters
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        stride: int,
        # conv + transformer parameters
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        # transformer parameters
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        p_stochastic_dropout: float,
        # partitioning parameters
        partition_size: int,
        dilation_size: Tuple[int, int],
        Flatten = False,
    ) -> None:
        super().__init__()

        layers: OrderedDict = OrderedDict()

        # convolutional layer
        layers["MBconv"] = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=stride,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        # attention layers, block -> dilation
        layers["window_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            dilation_size=dilation_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
            Flatten = Flatten,
        )
        layers["dilation_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="dilation",
            dilation_size=dilation_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
            Flatten = Flatten,
        )
        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        x = self.layers(x)
        return x


class DiWTBlock(nn.Module):
    def __init__(
        self,
        # conv parameters
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        # conv + transformer parameters
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        # transformer parameters
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        # partitioning parameters
        partition_size: int,
        input_dilation_size: Tuple[int, int],
        # number of layers
        n_layers: int,
        p_stochastic: List[float],
        Flatten = False,
    ) -> None:
        super().__init__()
        # if not len(p_stochastic) == n_layers:
        #     raise ValueError(f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}.")

        self.layers = nn.ModuleList()
        # account for the first stride of the first layer
        #self.dilation_size = _get_conv_output_shape(input_dilation_size, kernel_size=3, stride=2, padding=1)#input_dilation_size224*224/1024*1024

        for idx, p in enumerate(p_stochastic):
            stride = 1#2 if idx == 0 else 1
            self.layers += [
                DiWTLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    dilation_size=input_dilation_size,#下采样 (256,256)/(512,512)
                    p_stochastic_dropout=p,
                    Flatten = Flatten,
                ),
            ]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        for layer in self.layers:
            x = layer(x)
        return x

