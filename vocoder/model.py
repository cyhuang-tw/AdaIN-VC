"""Universal vocoder"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Vocoder(nn.Module):
    """Universal vocoding"""

    def __init__(
            self,
            sample_rate,
            frames_per_sample,
            frames_per_slice,
            mel_channels,
            conditioning_channels,
            embedding_dim,
            rnn_channels,
            fc_channels,
            bits,
            hop_length
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.frames_per_slice = frames_per_slice
        self.pad = (frames_per_sample - frames_per_slice) // 2
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2 ** bits
        self.hop_length = hop_length

        self.rnn1 = nn.GRU(
            mel_channels, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True
        )
        self.embedding = nn.Embedding(self.quantization_channels, embedding_dim)
        self.rnn2 = nn.GRU(embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, wavs, mels):
        """Generate waveform from mel spectrogram with teacher-forcing."""
        mel_embs, _ = self.rnn1(mels)
        mel_embs = mel_embs.transpose(1, 2)
        mel_embs = mel_embs[:, :, self.pad : self.pad + self.frames_per_slice]

        conditions = F.interpolate(mel_embs, scale_factor=float(self.hop_length))
        conditions = conditions.transpose(1, 2)

        wav_embs = self.embedding(wavs)
        wav_outs, _ = self.rnn2(torch.cat((wav_embs, conditions), dim=2))

        wav_outs = F.relu(self.fc1(wav_outs))
        wav_outs = self.fc2(wav_outs)

        return wav_outs

    @torch.jit.export
    def generate(self, mels: List[Tensor]) -> List[Tensor]:
        """Generate waveform from mel spectrogram.
        Args:
            mels: list of tensor of shape (mel_len, mel_channels)
        Returns:
            wavs: list of tensor of shape (wav_len)
        """

        # mels: List[(mel_len, mel_channels), ...]
        batch_size = len(mels)
        device = mels[0].device

        mel_lens = [len(mel) for mel in mels]
        wav_lens = [mel_len * self.hop_length for mel_len in mel_lens]
        max_mel_len = max(mel_lens)
        max_wav_len = max_mel_len * self.hop_length

        pad_mels = pad_sequence(mels, batch_first=True)
        pack_mels = pack_padded_sequence(
            pad_mels, torch.tensor(mel_lens), batch_first=True, enforce_sorted=False
        )
        pack_mel_embs, _ = self.rnn1(pack_mels)
        mel_embs, _ = pad_packed_sequence(pack_mel_embs, batch_first=True)

        # mel_embs: (batch, embedding_dim, max_mel_len)
        mel_embs = mel_embs.transpose(1, 2)

        # conditions: (batch, embedding_dim, max_wav_len)
        conditions = F.interpolate(mel_embs, scale_factor=float(self.hop_length))
        # conditions: (batch, max_wav_len, embedding_dim)
        conditions = conditions.transpose(1, 2)

        hid = torch.zeros(1, batch_size, self.rnn_channels, device=device)
        wav = torch.full(
            (batch_size,), self.quantization_channels // 2, dtype=torch.long, device=device,
        )
        wavs = torch.empty(batch_size, max_wav_len, dtype=torch.float, device=device,)

        for i, condition in enumerate(tqdm(torch.unbind(conditions, dim=1))):
            wav_emb = self.embedding(wav)
            wav_rnn_input = torch.cat((wav_emb, condition), dim=1).unsqueeze(1)
            _, hid = self.rnn2(wav_rnn_input, hid)

            logit = F.relu(self.fc1(hid.squeeze(0)))
            logit = self.fc2(logit)

            posterior = F.softmax(logit, dim=1)
            wav = torch.multinomial(posterior, 1).squeeze(1)
            wavs[:, i] = 2 * wav / (self.quantization_channels - 1.0) - 1.0

        mu = self.quantization_channels - 1
        wavs = torch.true_divide(torch.sign(wavs), mu) * (
            (1 + mu) ** torch.abs(wavs) - 1
        )
        wavs = [
            wav[:length] for wav, length in zip(torch.unbind(wavs, dim=0), wav_lens)
        ]

        return wavs
