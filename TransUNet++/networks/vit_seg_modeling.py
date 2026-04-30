
"""
TransUNet++ — vit_seg_modeling.py
"""

import copy
import logging
import math
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.nn import (CrossEntropyLoss, Dropout, LayerNorm, Linear, Softmax)
from torch.nn.modules.utils import _pair

from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .vit_seg_modeling_nested import UNetPlusPlusDecoder

logger = logging.getLogger(__name__)

ATTENTION_Q    = "MultiHeadDotProductAttention_1/query"
ATTENTION_K    = "MultiHeadDotProductAttention_1/key"
ATTENTION_V    = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT  = "MultiHeadDotProductAttention_1/out"
FC_0           = "MlpBlock_3/Dense_0"
FC_1           = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM       = "LayerNorm_2"


def np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


# ── Transformer sub-modules ──────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis           = vis
        self.num_heads     = config.transformer["num_heads"]
        self.head_size     = int(config.hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key   = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out   = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax      = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores  = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        probs   = self.softmax(scores)
        weights = probs if self.vis else None
        probs   = self.attn_dropout(probs)

        ctx = torch.matmul(probs, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size()[:-2] + (self.all_head_size,))
        out = self.proj_dropout(self.out(ctx))
        return out, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1  = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2  = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act  = ACT2FN["gelu"]
        self.drop = Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Embeddings(nn.Module):
    """Patch + position embeddings, with optional ResNetV2 hybrid backbone."""

    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.config = config
        img_size    = _pair(img_size)
        self.hybrid = False

        if config.patches.get("grid") is not None:
            grid_size       = config.patches["grid"]
            patch_size      = (img_size[0] // 16 // grid_size[0],
                               img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches       = ((img_size[0] // patch_size_real[0]) *
                               (img_size[1] // patch_size_real[1]))
            self.hybrid       = True
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 32
        else:
            patch_size = _pair(config.patches["size"])
            n_patches  = ((img_size[0] // patch_size[0]) *
                          (img_size[1] // patch_size[1]))

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = None
        if self.hybrid:
            x, features = self.hybrid_model(x)

        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)

        pos = self.position_embeddings
        if x.shape[1] != pos.shape[1]:
            pos = pos.transpose(1, 2)
            pos = F.interpolate(pos, size=x.shape[1], mode="linear", align_corners=False)
            pos = pos.transpose(1, 2)

        x = self.dropout(x + pos)
        return x, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.hidden_size    = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm       = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn            = Mlp(config)
        self.attn           = Attention(config, vis)

    def forward(self, x):
        h = x
        x, w = self.attn(self.attention_norm(x))
        x = x + h
        h = x
        x = self.ffn(self.ffn_norm(x)) + h
        return x, w

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            def _w(key): return np2th(weights[pjoin(ROOT, key)])

            self.attn.query.weight.copy_(_w(f"{ATTENTION_Q}/kernel").view(self.hidden_size, self.hidden_size).t())
            self.attn.key.weight.copy_(  _w(f"{ATTENTION_K}/kernel").view(self.hidden_size, self.hidden_size).t())
            self.attn.value.weight.copy_(_w(f"{ATTENTION_V}/kernel").view(self.hidden_size, self.hidden_size).t())
            self.attn.out.weight.copy_(  _w(f"{ATTENTION_OUT}/kernel").view(self.hidden_size, self.hidden_size).t())

            self.attn.query.bias.copy_(_w(f"{ATTENTION_Q}/bias").view(-1))
            self.attn.key.bias.copy_(  _w(f"{ATTENTION_K}/bias").view(-1))
            self.attn.value.bias.copy_(_w(f"{ATTENTION_V}/bias").view(-1))
            self.attn.out.bias.copy_(  _w(f"{ATTENTION_OUT}/bias").view(-1))

            self.ffn.fc1.weight.copy_(_w(f"{FC_0}/kernel").t())
            self.ffn.fc2.weight.copy_(_w(f"{FC_1}/kernel").t())
            self.ffn.fc1.bias.copy_(  _w(f"{FC_0}/bias").t())
            self.ffn.fc2.bias.copy_(  _w(f"{FC_1}/bias").t())

            self.attention_norm.weight.copy_(_w(f"{ATTENTION_NORM}/scale"))
            self.attention_norm.bias.copy_(  _w(f"{ATTENTION_NORM}/bias"))
            self.ffn_norm.weight.copy_(_w(f"{MLP_NORM}/scale"))
            self.ffn_norm.bias.copy_(  _w(f"{MLP_NORM}/bias"))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis          = vis
        self.layer        = nn.ModuleList(
            [copy.deepcopy(Block(config, vis)) for _ in range(config.transformer["num_layers"])]
        )
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, w = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(w)
        return self.encoder_norm(hidden_states), attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder    = Encoder(config, vis)

    def forward(self, x):
        emb, features    = self.embeddings(x)
        encoded, weights = self.encoder(emb)
        return encoded, weights, features


# ── Decoder components ────────────────────────────────────────────────────────

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0,
                 stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                         stride=stride, padding=padding,
                         bias=not use_batchnorm)
        bn   = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels,
                                kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels,
                                kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.up    = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, upsampling=1):
        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size,
                         padding=kernel_size // 2)
        up   = (nn.UpsamplingBilinear2d(scale_factor=upsampling)
                if upsampling > 1 else nn.Identity())
        super().__init__(conv, up)


class DecoderCup(nn.Module):
    """Standard U-Net style decoder (original TransUNet)."""

    def __init__(self, config):
        super().__init__()
        self.config    = config
        head_channels  = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels,
                                    kernel_size=3, padding=1)
        dec_ch      = config.decoder_channels
        in_channels = [head_channels] + list(dec_ch[:-1])
        out_channels = dec_ch

        n_skip = getattr(config, "n_skip", 0)
        if n_skip != 0:
            skip_channels = list(config.skip_channels)
            for i in range(4 - n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        self.blocks = nn.ModuleList([
            DecoderBlock(ic, oc, sc)
            for ic, oc, sc in zip(in_channels, out_channels, skip_channels)
        ])
        self.n_skip = n_skip

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, block in enumerate(self.blocks):
            skip = (features[i]
                    if features is not None and i < self.n_skip
                    else None)
            x = block(x, skip=skip)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    TransUNet / TransUNet++.

    Standard mode  (use_nested_decoder=False):
        forward() → logits  tensor  (B, n_classes, H, W)

    Nested mode    (use_nested_decoder=True):
        forward() → (logits, [aux_logit_0, aux_logit_1, aux_logit_2])
        aux channels (R50 width_factor=1): [128, 256, 512]
    """

    def __init__(self, config, img_size=224,
                 num_classes=21843, zero_head=False, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head   = zero_head
        self.classifier  = config.classifier
        self.config      = config

        self.transformer = Transformer(config, img_size, vis)

        self.use_nested = getattr(config, "use_nested_decoder", False)
        if self.use_nested:
            self.decoder = UNetPlusPlusDecoder(config)

            # Aux head input channels match UNetPlusPlusDecoder.CH:
            #   x30 → CH[3] = 128,  x20 → CH[2] = 256,  x10 → CH[1] = 512
            aux_in_ch = [
                UNetPlusPlusDecoder.CH[3],   # 128
                UNetPlusPlusDecoder.CH[2],   # 256
                UNetPlusPlusDecoder.CH[1],   # 512
            ]
            self.aux_heads = nn.ModuleList([
                SegmentationHead(
                    in_channels=ch,
                    out_channels=config.n_classes,
                    kernel_size=3,
                )
                for ch in aux_in_ch
            ])

            # Primary head reads CH[4] = 64
            primary_in_ch = UNetPlusPlusDecoder.CH[4]   # 64
        else:
            self.decoder      = DecoderCup(config)
            primary_in_ch     = config.decoder_channels[-1]

        self.segmentation_head = SegmentationHead(
            in_channels=primary_in_ch,
            out_channels=config.n_classes,
            kernel_size=3,
        )

        if zero_head:
            nn.init.zeros_(self.segmentation_head[0].weight)
            nn.init.zeros_(self.segmentation_head[0].bias)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        encoded, attn_weights, features = self.transformer(x)

        if self.use_nested:
            final_dec, aux_outputs = self.decoder(encoded, features)
            logits     = self.segmentation_head(final_dec)
            aux_logits = [head(aux) for head, aux in zip(self.aux_heads, aux_outputs)]
            return logits, aux_logits
        else:
            dec    = self.decoder(encoded, features)
            logits = self.segmentation_head(dec)
            return logits

    def load_from(self, weights):
        if weights is None:
            print("load_from: no weights provided, skipping.")
            return

        with torch.no_grad():
            if "embedding/kernel" in weights:
                self.transformer.embeddings.patch_embeddings.weight.copy_(
                    np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(
                    np2th(weights["embedding/bias"]))

            if "Transformer/encoder_norm/scale" in weights:
                self.transformer.encoder.encoder_norm.weight.copy_(
                    np2th(weights["Transformer/encoder_norm/scale"]))
                self.transformer.encoder.encoder_norm.bias.copy_(
                    np2th(weights["Transformer/encoder_norm/bias"]))

            if "Transformer/posembed_input/pos_embedding" in weights:
                posemb     = np2th(weights["Transformer/posembed_input/pos_embedding"])
                posemb_new = self.transformer.embeddings.position_embeddings
                if posemb.size() == posemb_new.size():
                    posemb_new.copy_(posemb)
                else:
                    logger.info("Resizing position embeddings: %s → %s",
                                posemb.size(), posemb_new.size())
                    ntok_new = posemb_new.size(1)
                    if self.classifier == "token":
                        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    else:
                        posemb_tok, posemb_grid = posemb[:, :0], posemb[0, :]

                    gs_old = int(np.sqrt(posemb_grid.size(0)))
                    gs_new = int(np.sqrt(ntok_new))
                    logger.info("Grid size: %d → %d", gs_old, gs_new)
                    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1)
                    zoom        = (gs_new / gs_old, gs_new / gs_old, 1)
                    posemb_grid = ndimage.zoom(posemb_grid[0], zoom, order=1)
                    posemb_grid = torch.from_numpy(posemb_grid).unsqueeze(0)
                    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                    posemb      = torch.cat([posemb_tok, posemb_grid], dim=1)
                    posemb_new.copy_(posemb)

            for n_block, layer_block in enumerate(self.transformer.encoder.layer):
                layer_block.load_from(weights, n_block=n_block)

            if (self.transformer.embeddings.hybrid and
                    hasattr(self.transformer.embeddings, "hybrid_model")):
                self.transformer.embeddings.hybrid_model.load_from(weights)


# ── Config registry ───────────────────────────────────────────────────────────

CONFIGS = {
    "ViT-B_16":                      configs.get_b16_config(),
    "ViT-B_32":                      configs.get_b32_config(),
    "ViT-L_16":                      configs.get_l16_config(),
    "ViT-L_32":                      configs.get_l32_config(),
    "ViT-H_14":                      configs.get_h14_config(),
    "R50-ViT-B_16":                  configs.get_r50_b16_config(),
    "R50-ViT-B_16-Plus":             configs.get_r50_b16_plus_config(),
    "R50-ViT-L_16":                  configs.get_r50_l16_config(),
    "ConvNeXt-ViT-B_16":             configs.get_convnext_b16_config(),
    "ConvNeXt-ViT-B_16-Plus":        configs.get_convnext_plus_b16_config(),
    "EfficientNet-B3-ViT-B_16":      configs.get_efficientnet_b3_config(),
    "EfficientNet-B3-ViT-B_16-Plus": configs.get_efficientnet_b3_plus_config(),
    "EfficientNet-B4-ViT-B_16":      configs.get_efficientnet_b4_config(),
    "testing":                       configs.get_testing(),
}
