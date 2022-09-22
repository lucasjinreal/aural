import argparse
from json import encoder
import logging
import numpy as np
import torch
from aural.modeling.beamsearch import FastBeamSearch, GreedySearch, ModifiedBeamSearch
from alfred import logger as logging

from aural.modeling.encoder import RNNEncoder
from aural.modeling.decoder import Decoder
from aural.modeling.joiner import Joiner

import argparse
import logging
from typing import List
import kaldifeat
import sentencepiece as spm
import torch
import torchaudio

torch.set_grad_enabled(False)

"""
Usage:
  python demo_file.py \
   --bpe-model-filename ./data/lang_bpe_500/bpe.model \
   --encoder-param-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
   --encoder-bin-filename ./lstm_transducer_stateless2/exp/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
   --decoder-param-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
   --decoder-bin-filename ./lstm_transducer_stateless2/exp/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
   --joiner-param-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param \
   --joiner-bin-filename ./lstm_transducer_stateless2/exp/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin \
   ./test_wavs/1089-134686-0001.wav
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe-model-filename",
        type=str,
        help="Path to bpe.model",
    )
    parser.add_argument(
        "--encoder-filename",
        type=str,
        help="Path to encoder.ncnn.param",
    )
    parser.add_argument(
        "--decoder-filename",
        type=str,
        help="Path to decoder.ncnn.param",
    )
    parser.add_argument(
        "--joiner-filename",
        type=str,
        help="Path to joiner.ncnn.param",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to foo.wav",
    )
    return parser.parse_args()


class Model:
    def __init__(self, args):
        self.init_encoder(args)
        self.init_decoder(args)
        self.init_joiner(args)

    def init_encoder(self, args):
        encoder_net = RNNEncoder()
        self.encoder_net = encoder_net

    def init_decoder(self, args):
        decoder_net = Decoder()
        self.decoder_net = decoder_net

    def init_joiner(self, args):
        joiner_net = Joiner()  
        self.joiner_net = joiner_net

    def run_encoder(self, x, states):
        x_lens = torch.tensor([x.size(0)], dtype=torch.float32)
        encoder_out, encoder_out_lens, hx, cx = self.encoder_net(x, x_lens, states[0], states[1])
        return encoder_out, encoder_out_lens, hx, cx

    def run_decoder(self, decoder_input):
        assert decoder_input.dtype == torch.int32
        decoder_out = self.decoder_net(decoder_input)
        return decoder_out

    def run_joiner(self, encoder_out, decoder_out):
        joiner_out = self.joiner_net(encoder_out, decoder_out)
        return joiner_out


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. " f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


def greedy_search(model: Model, encoder_out: torch.Tensor):
    assert encoder_out.ndim == 2
    T = encoder_out.size(0)
    context_size = 2
    blank_id = 0  # hard-code to 0
    hyp = [blank_id] * context_size

    decoder_input = torch.tensor(hyp, dtype=torch.int32)  # (1, context_size)

    decoder_out = model.run_decoder(decoder_input).squeeze(0)
    #  print(decoder_out.shape)  # (512,)
    for t in range(T):
        encoder_out_t = encoder_out[t]
        joiner_out = model.run_joiner(encoder_out_t, decoder_out)
        #  print(joiner_out.shape) # [500]
        y = joiner_out.argmax(dim=0).tolist()
        if y != blank_id:
            hyp.append(y)
            decoder_input = hyp[-context_size:]
            decoder_input = torch.tensor(decoder_input, dtype=torch.int32)
            decoder_out = model.run_decoder(decoder_input).squeeze(0)
    return hyp[context_size:]


def main():
    args = get_args()
    logging.info(vars(args))

    model = Model(args)

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model_filename)

    sound_file = args.sound_filename
    sample_rate = 16000

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    fbank = kaldifeat.Fbank(opts)
    logging.info('FBank feat will run on CPU.')

    logging.info(f"Reading sound files: {sound_file}")
    wave_samples = read_sound_files(
        filenames=[sound_file],
        expected_sample_rate=sample_rate,
    )[0]

    logging.info("Decoding started")
    features = fbank(wave_samples)

    num_encoder_layers = 12
    d_model = 512
    rnn_hidden_size = 1024

    states = (
        torch.zeros(num_encoder_layers, d_model),
        torch.zeros(
            num_encoder_layers,
            rnn_hidden_size,
        ),
    )

    encoder_out, encoder_out_lens, hx, cx = model.run_encoder(features, states)
    hyp = greedy_search(model, encoder_out)
    logging.info(sound_file)
    logging.info(sp.decode(hyp))


if __name__ == "__main__":
    main()
