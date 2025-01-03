import torch.nn as nn

"""
This implementation is based on the SparK (Sparse masKed modeling) approach
from the paper "Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling"
by Tian et al. (https://arxiv.org/pdf/2301.03580.pdf).

Modifications:
- Adapted the _get_active_ex_or_ii function to work with spatial variables
- Integrated the sparse masking strategy with the ConvNeXt architecture
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath

_cur_active: torch.Tensor = None

def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    cur_H, cur_W = _cur_active.shape[-2:]

    if H != cur_H or W != cur_W:
        scale_factor = cur_H // H
        active_ex = F.max_pool2d(_cur_active, kernel_size=scale_factor, stride=scale_factor)
        active_ex = (active_ex > 0.5).float()
    else:
        active_ex = _cur_active

    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)

    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`

    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 4: # BHWC or BCHW
            if self.data_format == "channels_last": # BHWC
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[1], W=x.shape[2], returning_active_ex=False)
                    nc = x[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else:       # channels_first, BCHW
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
                    bhwc = x.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    x = torch.zeros_like(bhwc)
                    x[ii] = nc
                    return x.permute(0, 3, 1, 2)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                    return x
        else:           # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseConvNeXtLayerNorm, self).__repr__()[:-1] + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'


class SparseConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, sparse=True, ks=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=ks//2, groups=dim)  # depthwise conv
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sparse = sparse

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)            # GELU(0) == (0), so there is no need to mask x (no need to `x *= _get_active_ex_or_ii`)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.sparse:
            x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)

        x = input + self.drop_path(x)
        return x

    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f', sp={self.sparse})'


class SparseEncoder(nn.Module):
    def __init__(self, cnn, input_size, sbn=False, verbose=False):
        super(SparseEncoder, self).__init__()
        self.sp_cnn = SparseEncoder.dense_model_to_sparse(m=cnn, verbose=verbose, sbn=sbn)
        self.input_size, self.downsample_raito, self.enc_feat_map_chs = input_size, cnn.get_downsample_ratio(), cnn.get_feature_map_channels()

    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        oup = m
        if isinstance(m, nn.Conv2d):
            m: nn.Conv2d
            bias = m.bias is not None
            oup = SparseConv2d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool2d):
            m: nn.MaxPool2d
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool2d):
            m: nn.AvgPool2d
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m: nn.BatchNorm2d
            oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            m: nn.LayerNorm
            oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, (nn.Conv1d,)):
            raise NotImplementedError

        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup

    def forward(self, x):
        return self.sp_cnn(x, hierarchical=True)

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., global_pool='avg',
                 sparse=True,
                 ):
        super().__init__()
        self.dims: List[int] = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            SparseConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first", sparse=sparse)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                SparseConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first", sparse=sparse),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[SparseConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                      layer_scale_init_value=layer_scale_init_value, sparse=sparse) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.depths = depths

        self.apply(self._init_weights)
        if num_classes > 0:
            self.norm = SparseConvNeXtLayerNorm(dims[-1], eps=1e-6, sparse=False)  # final norm layer for LE/FT; should not be sparse
            self.fc = nn.Linear(dims[-1], num_classes)
        else:
            self.norm = nn.Identity()
            self.fc = nn.Identity()

            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def get_downsample_ratio(self) -> int:
        return 32

    def get_feature_map_channels(self) -> List[int]:
        return self.dims

    def forward(self, x, hierarchical=False):
        if hierarchical:
            ls = []
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                ls.append(x)
            return ls
        else:
            return self.fc(self.norm(x.mean([-2, -1]))) # (B, C, H, W) =mean=> (B, C) =norm&fc=> (B, NumCls)

    def get_classifier(self):
        return self.fc

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}, layer_scale_init_value={self.layer_scale_init_value:g}'

def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)

class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
        )

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, up_sample_ratio, out_chans=1, width=768, sbn=True):   # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(n + 1)] # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn2d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv2d(channels[-1], out_chans, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
