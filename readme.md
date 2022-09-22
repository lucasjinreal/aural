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


## Demo

1. `demo for English librispeech data`:

Download pretrained model [here](https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03/blob/main/exp/pretrained-iter-472000-avg-18.pt)


2. `demo for Chinese wenetspeech data`:

tbd



## References

1. TTS: https://github.com/MiniXC/LightningFastSpeech2
2. icefall: https://github.com/k2-fsa/icefall

## Copyright

all rights reserverd by Lucas Jin.
