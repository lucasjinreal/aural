import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from aural.utils.scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params, get_transducer_model
from aural.utils.util import str2bool
from alfred import logger as logging


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--bpe_model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )
    parser.add_argument(
        "--jit-trace",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.trace.
        It will generate 3 files:
         - encoder_jit_trace.pt
         - decoder_jit_trace.pt
         - joiner_jit_trace.pt
        Check ./jit_pretrained.py for how to use them.
        """,
    )
    parser.add_argument(
        "--pnnx",
        type=str2bool,
        default=True,
        help="""True to save a model after applying torch.jit.trace for later
        converting to PNNX. It will generate 3 files:
         - encoder_jit_trace-pnnx.pt
         - decoder_jit_trace-pnnx.pt
         - joiner_jit_trace-pnnx.pt
        """,
    )
    parser.add_argument("-p", "--pretrained_model", type=str, help="pretrained model")
    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; " "2 means tri-gram",
    )
    add_model_arguments(parser)
    return parser


def export_encoder_model_jit_trace(
    encoder_model: nn.Module,
    encoder_filename: str,
) -> None:
    """Export the given encoder model with torch.jit.trace()
    Note: The warmup argument is fixed to 1.
    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported model.
    """
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)
    states = encoder_model.get_init_states()

    traced_model = torch.jit.trace(encoder_model, (x, x_lens, states))
    traced_model.save(encoder_filename)
    logging.info(f"Saved to {encoder_filename}")

    o_f = str(encoder_filename).split(".")[0] + ".onnx"
    # torch.onnx.export(encoder_model, (x, x_lens, states), o_f)
    logging.info(f"Saved ONNX to {o_f}")


def export_decoder_model_jit_trace(
    decoder_model: nn.Module,
    decoder_filename: str,
) -> None:
    """Export the given decoder model with torch.jit.trace()
    Note: The argument need_pad is fixed to False.
    Args:
      decoder_model:
        The input decoder model
      decoder_filename:
        The filename to save the exported model.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)
    need_pad = torch.tensor([False])

    traced_model = torch.jit.trace(decoder_model, (y, need_pad))
    traced_model.save(decoder_filename)
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_jit_trace(
    joiner_model: nn.Module,
    joiner_filename: str,
) -> None:
    """Export the given joiner model with torch.jit.trace()
    Note: The argument project_input is fixed to True. A user should not
    project the encoder_out/decoder_out by himself/herself. The exported joiner
    will do that for the user.
    Args:
      joiner_model:
        The input joiner model
      joiner_filename:
        The filename to save the exported model.
    """
    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

    traced_model = torch.jit.trace(joiner_model, (encoder_out, decoder_out))
    traced_model.save(joiner_filename)
    logging.info(f"Saved to {joiner_filename}")


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    params = get_params()
    params.update(vars(args))
    params.update({"exp_dir": "weights"})
    params.exp_dir = Path(params.exp_dir)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    if params.pnnx:
        params.is_pnnx = params.pnnx
        logging.info("For PNNX")

    logging.info("About to create model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    model.to(device)
    model.load_state_dict(
        torch.load(args.pretrained_model, "cpu"),
        strict=False,
    )

    model.to("cpu")
    model.eval()
    logging.info("model loaded!")

    if params.pnnx:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace-pnnx.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace-pnnx.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace-pnnx.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    elif params.jit_trace is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    else:
        logging.info("Not using torchscript")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    main()
