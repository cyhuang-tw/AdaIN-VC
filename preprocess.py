import os
import argparse
import pickle
import random
from functools import partial
from multiprocessing import Pool

import json
import numpy as np

from vocoder.utils import load_wav, melspectrogram

def get_spectrogram(file_name, params):
    wav = load_wav(file_name, params["preprocessing"]["sample_rate"])
    wav = wav / np.abs(wav).max() * 0.999
    mel = melspectrogram(wav, sample_rate=params["preprocessing"]["sample_rate"],
                         preemph=params["preprocessing"]["preemph"],
                         num_mels=params["preprocessing"]["num_mels"],
                         num_fft=params["preprocessing"]["num_fft"],
                         min_level_db=params["preprocessing"]["min_level_db"],
                         hop_length=params["preprocessing"]["hop_length"],
                         win_length=params["preprocessing"]["win_length"],
                         fmin=params["preprocessing"]["fmin"])
    return (file_name, mel.astype(np.float32))

def main(data_dir, output_dir, config, n_test_speakers, valid_proportion, segment_size, train_samples):
    random.seed(1001)

    with open(config, 'r') as f:
        params = json.load(f)

    speaker_ids = sorted(os.listdir(data_dir))
    random.shuffle(speaker_ids)

    train_speaker_ids = speaker_ids[:-n_test_speakers]
    test_speaker_ids = speaker_ids[-n_test_speakers:]

    train_files, valid_files, test_files = [], [], []

    for speaker in train_speaker_ids:
        file_list = sorted([os.path.join(data_dir, speaker, x)\
                            for x in os.listdir(os.path.join(data_dir, speaker))])
        random.shuffle(file_list)
        valid_size = int(len(file_list) * valid_proportion)
        train_files += file_list[:-valid_size]
        valid_files += file_list[-valid_size:]

    for speaker in test_speaker_ids:
        file_list = sorted([os.path.join(data_dir, speaker, x)\
                            for x in os.listdir(os.path.join(data_dir, speaker))])
        test_files += file_list

    os.makedirs(output_dir, exist_ok=True)

    for file_type, file_list in zip(['train', 'valid', 'test'],\
                                    [train_files, valid_files, test_files]):
        print(f'Processing {file_type} data...')
        with Pool(8) as p:
            mel_list = p.map_async(partial(get_spectrogram, params=params), file_list).get()
        mel_list = filter(lambda x: len(x[1]) > segment_size, mel_list)
        mel_data = dict(mel_list)
        with open(os.path.join(output_dir, f'{file_type}.pkl'), 'wb') as f:
            pickle.dump(mel_data, f)

        if file_type == 'train':
            key_list = sorted(list(mel_data.keys()))
            sample_list = random.choices(range(len(key_list)), k=train_samples)
            samples = []
            for index in sample_list:
                t = random.randint(0, len(mel_data[key_list[index]]) - segment_size)
                samples.append((key_list[index], t))
            with open(os.path.join(output_dir, f'train_{segment_size}.json'), 'w') as f:
                json.dump(samples, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--config', type=str, default='vocoder/config.json')
    parser.add_argument('--n_test_speakers', type=int, default=20)
    parser.add_argument('--valid_proportion', type=float, default=0.1)
    parser.add_argument('--segment_size', type=int, default=128)
    parser.add_argument('--train_samples', type=int, default=1000000)
    main(**vars(parser.parse_args()))
