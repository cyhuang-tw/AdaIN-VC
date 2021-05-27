import argparse
import json
import os
from functools import partial
from uuid import uuid4

import librosa
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchaudio
from torch import Tensor
from tqdm.auto import tqdm

from data.wav2mel import Wav2Mel


def process_files(audio_file: str, wav2mel: nn.Module) -> Tensor:
    speech_tensor, sample_rate = torchaudio.load(audio_file)
    mel_tensor = wav2mel(speech_tensor, sample_rate)

    return mel_tensor


def main(data_dir: str, save_dir: str, segment: int):
    mp.set_sharing_strategy("file_system")
    os.makedirs(save_dir, exist_ok=True)
    wav2mel = Wav2Mel()
    file2mel = partial(process_files, wav2mel=wav2mel)

    meta_data = {}
    speakers = sorted(os.listdir(data_dir))

    for spk in tqdm(speakers):
        spk_dir = os.path.join(data_dir, spk)
        wav_files = librosa.util.find_files(spk_dir)
        mels = [file2mel(wav_file) for wav_file in wav_files]
        mels = list(filter(lambda x: x is not None and x.shape[-1] > segment, mels))
        rnd_paths = [f"{uuid4().hex}.pt" for _ in range(len(mels))]
        dummy = [
            torch.save(mel, os.path.join(save_dir, path))
            for (mel, path) in zip(mels, rnd_paths)
        ]
        meta_data[spk] = rnd_paths

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--segment", type=int, default=128)
    main(**vars(parser.parse_args()))
