from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm


def get_act(act: str) -> nn.Module:
    if act == "lrelu":
        return nn.LeakyReLU()
    return nn.ReLU()


class ConvBank(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_bank: int, bank_scale: int, act: str):
        super(ConvBank, self).__init__()
        self.conv_bank = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d((k // 2, k // 2 - 1 + k % 2)),
                    nn.Conv1d(c_in, c_out, kernel_size=k),
                )
                for k in range(bank_scale, n_bank + 1, bank_scale)
            ]
        )
        self.act = get_act(act)

    def forward(self, x: Tensor) -> Tensor:
        outs = [self.act(layer(x)) for layer in self.conv_bank]
        out = torch.cat(outs + [x], dim=1)
        return out


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor: int):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, in_width = x.size()
        channels = channels // self.scale_factor
        out_width = in_width * self.scale_factor
        x = x.contiguous().view(batch_size, channels, self.scale_factor, in_width)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, channels, out_width)
        return x


class AffineLayer(nn.Module):
    def __init__(self, c_cond: int, c_h: int):
        super(AffineLayer, self).__init__()
        self.c_h = c_h
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.linear_layer = nn.Linear(c_cond, c_h * 2)

    def forward(self, x: Tensor, x_cond: Tensor) -> Tensor:
        x_cond = self.linear_layer(x_cond)
        mean, std = x_cond[:, : self.c_h], x_cond[:, self.c_h :]
        mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)
        x = self.norm_layer(x)
        x = x * std + mean
        return x


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        bank_size: int,
        bank_scale: int,
        c_bank: int,
        n_conv_blocks: int,
        n_dense_blocks: int,
        subsample: List[int],
        act: str,
        dropout_rate: float,
    ):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = ConvBank(c_in, c_bank, bank_size, bank_scale, act)
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    nn.Conv1d(c_h, c_h, kernel_size=kernel_size),
                )
                for _ in range(n_conv_blocks)
            ]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub),
                )
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.second_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp: Tensor) -> Tensor:
        out = inp
        for idx, (first_layer, second_layer) in enumerate(
            zip(self.first_conv_layers, self.second_conv_layers)
        ):
            y = first_layer(out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = second_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[idx] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[idx], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp: Tensor) -> Tensor:
        out = inp
        for first_layer, second_layer in zip(
            self.first_dense_layers, self.second_dense_layers
        ):
            y = first_layer(out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = second_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_bank(x)
        out = self.in_conv_layer(out)
        out = self.act(out)
        out = self.conv_blocks(out)
        out = self.pooling_layer(out).squeeze(-1)
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out


class ContentEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        bank_size: int,
        bank_scale: int,
        c_bank: int,
        n_conv_blocks: int,
        subsample: List[int],
        act: str,
        dropout_rate: float,
    ):
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = ConvBank(c_in, c_bank, bank_size, bank_scale, act)
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    nn.Conv1d(c_h, c_h, kernel_size=kernel_size),
                )
                for _ in range(n_conv_blocks)
            ]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub),
                )
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.conv_bank(x)
        out = self.in_conv_layer(out)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        for idx, (first_layer, second_layer) in enumerate(
            zip(self.first_conv_layers, self.second_conv_layers)
        ):
            y = first_layer(out)
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = second_layer(y)
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = F.avg_pool1d(out, kernel_size=self.subsample[idx], ceil_mode=True)
            out = y + out
        mu = self.mean_layer(out)
        log_sigma = self.std_layer(out)
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_cond: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        n_conv_blocks: int,
        upsample: List[int],
        act: str,
        sn: bool,
        dropout_rate: float,
    ):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)),
                )
                for _ in range(n_conv_blocks)
            ]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(
                        (kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)
                    ),
                    nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size),
                )
                for _, up in zip(range(n_conv_blocks), self.upsample)
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.first_affine_layers = nn.ModuleList(
            [AffineLayer(c_cond, c_h) for _ in range(n_conv_blocks)]
        )
        self.second_affine_layers = nn.ModuleList(
            [AffineLayer(c_cond, c_h) for _ in range(n_conv_blocks)]
        )
        self.pixel_shuffle = nn.ModuleList(
            [PixelShuffle(scale_factor) for scale_factor in self.upsample]
        )
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        out = self.in_conv_layer(z)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        for idx, (
            first_conv_layer,
            second_conv_layer,
            first_affine_layer,
            second_affine_layer,
            pixel_shuffle,
        ) in enumerate(
            zip(
                self.first_conv_layers,
                self.second_conv_layers,
                self.first_affine_layers,
                self.second_affine_layers,
                self.pixel_shuffle,
            )
        ):
            y = first_conv_layer(out)
            y = self.norm_layer(y)
            y = first_affine_layer(y, cond)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = second_conv_layer(y)
            y = pixel_shuffle(y)
            y = self.norm_layer(y)
            y = second_affine_layer(y, cond)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + F.interpolate(
                out, scale_factor=float(self.upsample[idx]), mode="nearest"
            )
        out = self.out_conv_layer(out)
        return out


class AdaINVC(nn.Module):
    def __init__(self, config: Dict):
        super(AdaINVC, self).__init__()
        self.speaker_encoder = SpeakerEncoder(**config["SpeakerEncoder"])
        self.content_encoder = ContentEncoder(**config["ContentEncoder"])
        self.decoder = Decoder(**config["Decoder"])

    def forward(
        self, src: Tensor, tgt: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if tgt is None:
            emb = self.speaker_encoder(src)
        else:
            emb = self.speaker_encoder(tgt)
        mu, log_sigma = self.content_encoder(src)
        eps = torch.empty_like(log_sigma).normal_(0.0, 1.0)
        dec = self.decoder(mu + torch.exp(log_sigma / 2.0) * eps, emb)
        return mu, log_sigma, emb, dec

    @torch.jit.export
    def inference(self, src: Tensor, tgt: Tensor) -> Tensor:
        emb = self.speaker_encoder(tgt)
        mu, _ = self.content_encoder(src)
        dec = self.decoder(mu, emb)
        return dec
