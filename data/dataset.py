import json
import os
import random

import torch
from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    def __init__(self, data_dir, segment=128, n_uttrs=4):
        self.data_dir = data_dir
        self.meta_data = json.load(open(os.path.join(data_dir, "metadata.json"), "r"))
        self.id2spk = list(self.meta_data.keys())
        self.segment = segment
        self.n_uttrs = n_uttrs

    def __len__(self):
        return len(self.meta_data)  # num_speakers

    def __getitem__(self, index):
        spk = self.id2spk[index]
        mel_files = random.sample(self.meta_data[spk], k=self.n_uttrs)
        mels = [torch.load(os.path.join(self.data_dir, file)) for file in mel_files]
        starts = [random.randint(0, m.shape[-1] - self.segment) for m in mels]
        mels = torch.stack(
            [m[:, start : (start + self.segment)] for (m, start) in zip(mels, starts)]
        )
        return mels
