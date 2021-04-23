import argparse

import soundfile as sf
import torch
import torchaudio

from data import Wav2Mel


def main(
    model_path: str,
    vocoder_path: str,
    source: str,
    target: str,
    output: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path).to(device)
    vocoder = torch.jit.load(vocoder_path).to(device)
    wav2mel = Wav2Mel()

    src, src_sr = torchaudio.load(source)
    tgt, tgt_sr = torchaudio.load(target)

    src = wav2mel(src, src_sr)[None, :].to(device)
    tgt = wav2mel(tgt, tgt_sr)[None, :].to(device)

    cvt = model.inference(src, tgt)

    with torch.no_grad():
        wav = vocoder.generate([cvt.squeeze(0).data.T])

    wav = wav[0].data.cpu().numpy()
    sf.write(output, wav, wav2mel.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("vocoder_path", type=str)
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("output", type=str)
    main(**vars(parser.parse_args()))
