import argparse
from json import encoder
from struct import pack
import numpy as np
import torch
from aural.modeling.post.beamsearch import (
    FastBeamSearch,
    GreedySearch,
    ModifiedBeamSearch,
)
import k2
from alfred import logger as logging
import argparse
from typing import List
import kaldifeat
import sentencepiece as spm
import torch
import torchaudio
from aural.modeling.meta_arch.lstm_transducer import build_lstm_transducer_model
from alfred import print_shape
from aural.modeling.post.geedysearch import (
    greedy_search_batch,
    greedy_search_single_batch,
    GreedySearchOffline,
)
from aural.modeling.meta_arch.conformer_transducer import (
    build_conformer_transducer_model,
    get_default_params,
)
from aural.utils.lexicon import Lexicon


torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bpe_model", type=str, help="Path to bpe.model")
    parser.add_argument("-p", "--pretrained_model", type=str, help="pretrained model")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="data/test_data/libri_reaer.wav",
        help="foo.wav",
    )
    return parser.parse_args()


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. " f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


def greedy_search_simple(model, encoder_out: torch.Tensor):
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

    if "bpe" in args.bpe_model:
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)
    else:
        lexions = Lexicon(args.bpe_model)
        # vc  = max(token_table) + 1
        print(len(lexions.tokens))

    sound_file = args.file
    sample_rate = 16000

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    fbank = kaldifeat.Fbank(opts)
    logging.info("FBank feat will run on CPU.")

    logging.info(f"Reading sound files: {sound_file}")
    wave_samples = read_sound_files(
        filenames=[sound_file],
        expected_sample_rate=sample_rate,
    )[0]

    logging.info("Decoding started")
    features = fbank(wave_samples)
    features = features.unsqueeze(0)

    if "lstm" in args.pretrained_model:
        logging.info("using lstm model.")
        rnn_hidden_size = 1024
        num_encoder_layers = 12
        d_model = 512
        asr_model = build_lstm_transducer_model(sp)
    else:
        logging.info("using the Conformer model.")
        params = get_default_params()
        asr_model = build_conformer_transducer_model(lexions, params)

    asr_model.load_state_dict(
        torch.load(args.pretrained_model, map_location="cpu")["model"]
    )
    asr_model.eval()
    logging.info("asr model loaded!")

    print_shape(features)

    if "lstm" in args.pretrained_model:
        states = (
            torch.zeros(num_encoder_layers, features.size(0), d_model),
            torch.zeros(
                num_encoder_layers,
                features.size(0),
                rnn_hidden_size,
            ),
        )
        encoder_out, encoder_out_lens, hx, cx = asr_model.run_encoder(features, states)
        # hyp = greedy_search(asr_model, encoder_out)
        hyp = greedy_search_single_batch(asr_model, encoder_out, encoder_out_lens)
        logging.info(sound_file)

        if "bpe" in args.bpe_model:
            res = sp.decode(hyp)
        else:
            print(lexions.token_table)
            print(hyp)
            res = [lexions.token_table[i] for i in hyp]
            res = ["".join(r) for r in res]
        logging.info(res)
    else:
        logging.info("runing on conformer..")
        encoder_out, encoder_out_lens = asr_model.run_encoder(features)
        tokens = greedy_search_single_batch(asr_model, encoder_out, encoder_out_lens)
        logging.info(sound_file)

        if "bpe" in args.bpe_model:
            res = sp.decode(tokens)
        else:
            res = [lexions.token_table[i] for i in tokens]
            res = "".join(res) 
        logging.info(res)


if __name__ == "__main__":
    main()
