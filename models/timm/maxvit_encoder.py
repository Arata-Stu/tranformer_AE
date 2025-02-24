from typing import Tuple, Union, List
import torch
from torch import nn
from functools import partial
from omegaconf import DictConfig

from .layers import get_norm_act_layer, create_conv2d
from .layers import to_2tuple


from .maxvit import (
    named_apply,
    _init_conv,
    _init_transformer,
    MbConvBlock,
    PartitionAttentionCl,

    )

class Stem(nn.Module):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            padding: str = '',
            bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs[-1]
        self.stride = 2

        self.conv1 = create_conv2d(in_chs, out_chs[0], kernel_size, stride=2, padding=padding, bias=bias)
        self.norm1 = norm_act_layer(out_chs[0])
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, padding=padding, bias=bias)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x
    

class MaxxVitEncoderBlock(nn.Module):
    """ MaxVit conv, window partition + FFN , grid partition + FFN
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            stride: int = 1,
            conv_cfg: DictConfig = None,
            transformer_cfg: DictConfig = None,
            drop_path: float = 0.,
    ):
        super().__init__()
        
        conv_cls = MbConvBlock
        self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)

        attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
        partition_layer = PartitionAttentionCl
        self.attn_block = partition_layer(**attn_kwargs)
        self.attn_grid = partition_layer(partition_type='grid', **attn_kwargs)

    def init_weights(self, scheme=''):
        if self.attn_block is not None:
            named_apply(partial(_init_transformer, scheme=scheme), self.attn_block)
        named_apply(partial(_init_transformer, scheme=scheme), self.attn_grid)
        named_apply(partial(_init_conv, scheme=scheme), self.conv)

    def forward(self, x):
        # NCHW format
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # to NHWC (channels-last)
        x = self.attn_block(x)
        x = self.attn_grid(x)
        x = x.permute(0, 3, 1, 2)  # back to NCHW
        return x
    

class MaxxVitEncoderStage(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 2,
            depth: int = 2,
            transformer_cfg: DictConfig = None,
            conv_cfg: DictConfig = None,
            drop_path: Union[float, List[float]] = 0.,
    ):
        super().__init__()
        self.grad_checkpointing = False
        blocks = []
        ## depth: number of blocks in the stage
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            
            blocks += [MaxxVitEncoderBlock(
                in_chs,
                out_chs,
                stride=block_stride,
                conv_cfg=conv_cfg,
                transformer_cfg=transformer_cfg,
                drop_path=drop_path[i],
            )]
            
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x
    

class MaxxVitEncoder(nn.Module):
    """ CoaTNet + MaxVit base model.

    Highly configurable for different block compositions, tensor layouts, pooling types.
    """

    def __init__(
            self,
            maxvit_cfg: DictConfig,
            img_size: Union[int, Tuple[int, int]] = 224,
            in_chans: int = 3,
            drop_path_rate: float = 0.,
    ):
        super().__init__()
        
        stem_cfg = maxvit_cfg.stem

        self.stem = Stem(
            in_chs=in_chans,
            out_chs=stem_cfg.out_chs,
            padding=stem_cfg.conv.padding,
            bias=stem_cfg.conv.bias,
            act_layer=stem_cfg.conv.act_layer,
            norm_layer=stem_cfg.conv.norm_layer,
            norm_eps=stem_cfg.conv.norm_eps,
        )

        stage_cfg = maxvit_cfg.stage

        transformer_cfg = stage_cfg.transformer
        img_size = to_2tuple(img_size)
        partition_size = img_size[0] // transformer_cfg.partition_ratio, img_size[1] // transformer_cfg.partition_ratio
        transformer_cfg.window_size = partition_size 
        transformer_cfg.grid_size = partition_size
        self.feature_info = []

        stride = self.stem.stride
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])

        num_stages = stage_cfg.num_stages
        assert len(stage_cfg.num_blocks) == num_stages
        num_blocks = to_2tuple(stage_cfg.num_blocks)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(num_blocks)).split(num_blocks)]
        in_chs = self.stem.out_chs
        stages = []
        for i in range(num_stages):
            stage_stride = 2
            out_chs = stage_cfg.embed_dim * stage_cfg.dim_multiplier[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages += [MaxxVitEncoderStage(
                in_chs,
                out_chs,
                depth=num_blocks[i],
                conv_cfg=stage_cfg.conv,
                transformer_cfg=transformer_cfg,
                drop_path=dpr[i],
            )]
            stride *= stage_stride
            in_chs = out_chs
            self.feature_info += [dict(num_chs=out_chs, reduction=stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert maxvit_cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        if maxvit_cfg.weight_init:
            named_apply(partial(self._init_weights, scheme=maxvit_cfg.weight_init), self)

    def _init_weights(self, module, name, scheme=''):
        if hasattr(module, 'init_weights'):
            try:
                module.init_weights(scheme=scheme)
            except TypeError:
                module.init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x
