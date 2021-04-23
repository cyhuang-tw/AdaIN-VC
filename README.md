# AdaIN-VC

This is an unofficial implementation of the paper [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742) modified from the official one.

## Dependencies

- Python >= 3.6
- torch >= 1.7.0
- torchaudio >= 0.7.0
- numpy >= 1.16.0
- librosa >= 0.6.3

## Differences from the official implementation

The main difference from the official implementation is the use of a neural vocoder, which greatly improves the audio quality.
I adopted universal vocoder, whose code was from [yistLin/universal-vocoder](https://github.com/yistLin/universal-vocoder) and checkpoint will be available soon.
Besides, this implementation supports torch.jit, so the full model can be loaded with simply one line:

```python
model = torch.jit.load(model_path)
```

Pre-trained models are available [here](https://drive.google.com/drive/folders/1MacKgXGA4Ad0O_c6W5MlkZMG0B8IzaM-?usp=sharing).

## Preprocess

The code `preprocess.py` extracts features from raw audios.

```bash
python preprocess.py <data_dir> <save_dir> [--segment seg_len]
```

- **data_dir**: The directory of speakers.
- **save_dir**: The directory to save the processed files.
- **seg_len**: The length of segments for training.

## Training

```bash
python train.py <config_file> <data_dir> <save_dir> [--n_steps steps] [--save_steps save] [--log_steps log] [--n_spks spks] [--n_uttrs uttrs]
```

- **config_file**: The config file for AdaIN-VC.
- **data_dir**: The directory of processed files given by `preprocess.py`.
- **save_dir**: The directory to save the model.
- **steps**: The number of steps for training.
- **save**: To save the model every <em>save</em> steps.
- **log**: To record training information every <em>log</em> steps.
- **spks**: The number of speakers in the batch.
- **uttrs**: The number of utterances for each speaker in the batch.

## Inference

You can use `inference.py` to perform one-shot voice conversion.
The pre-trained model will be available soon.

```bash
python inference.py <model_path> <vocoder_path> <source> <target> <output>
```

- **model_path**: The path of the model file.
- **vocoder_path**: The path of the vocoder file.
- **source**: The utterance providing linguistic content.
- **target**: The utterance providing target speaker timbre.
- **output**: The converted utterance.

## Reference

Please cite the paper if you find AdaIN-VC useful.

```bib
@article{chou2019one,
  title={One-shot voice conversion by separating speaker and content representations with instance normalization},
  author={Chou, Ju-chieh and Yeh, Cheng-chieh and Lee, Hung-yi},
  journal={arXiv preprint arXiv:1904.05742},
  year={2019}
}
```
