# Aural

**Aural** is a project which focusing on training a ASR model. Then using the output model to driven more applications which need ASR.

For instance, we can using aural model to listen command, just by saying, let your AI can understand what you saying.

> still WIP.

We will exporting the ASR model and inference via WNNX.

> Aural is build based on **k2**, and mostly reconstructed from icefall. 

Be note:

> Due to it need kaldifeat and kaldialign for data process, Windows not support for now, working on it.

## Install

Before runing `aural`, there are some deps need to install:

```
pip install pydub
pip install kaldifeat
pip install kaldialign
pip install alfred-py
pip install sentencepiece
```

In addition, `k2` is better install from source:

```
git clone https://github.com/k2-fsa/k2/
cd k2
python setup.py install
```


## Demo

1. `demo for English librispeech data`:

Before training, we can test on the correctness of the mode, using a pretrained model which comes from icefall:

```
git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/
```

this should download `token`, `bpe`, `pretrained_model` which we need enough to inference it. Be note that, `Aural` doesn't need any `icefall` code, it can run itself, compatible with `icefall` trained model.

Move your cloned file to `weights`, then run:

```
python demo_file.py --bpe_model weights/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/data/lang_bpe_500/bpe.model --pretrained_model weights/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/exp/pretrained.pt
```


2. `demo for Chinese wenetspeech data`:

Download wenet pretrained model:

```
git clone https://huggingface.co/csukuangfj/icefall-asr-wenetspeech-lstm-transducer-stateless-2022-09-19
```


## Export JIT ONNX

To export the model for JIT for onnx:

```
python export.py --pretrained_model weights/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/exp/pretrained.pt --bpe_model weights/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18/data/lang_bpe_500/bpe.model
```


## References

1. TTS: https://github.com/MiniXC/LightningFastSpeech2
2. icefall: https://github.com/k2-fsa/icefall

## Copyright

all rights reserverd by Lucas Jin.
