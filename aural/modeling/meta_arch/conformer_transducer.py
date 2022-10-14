import argparse
from aural.utils.util import AttributeDict
from .transducer import Transducer
from ..encoders.rnn import RNNEncoder, RNN
from ..encoders.conformer import Conformer
from ..decoders.decoder import Decoder
from ..post.joiner import Joiner
from aural.utils.util import str2bool
from aural.utils.lexicon import Lexicon
from alfred import logger


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dynamic-chunk-training",
        type=str2bool,
        default=False,
        help="""Whether to use dynamic_chunk_training, if you want a streaming
        model, this requires to be True.
        """,
    )
    parser.add_argument(
        "--causal-convolution",
        type=str2bool,
        default=False,
        help="""Whether to use causal convolution, this requires to be True when
        using dynamic_chunk_training.
        """,
    )
    parser.add_argument(
        "--short-chunk-size",
        type=int,
        default=25,
        help="""Chunk length of dynamic training, the chunk size would be either
        max sequence length of current batch or uniformly sampled from (1, short_chunk_size).
        """,
    )
    parser.add_argument(
        "--num-left-chunks",
        type=int,
        default=4,
        help="How many left context can be seen in chunks when calculating attention.",
    )


def get_default_params():
    return AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "encoder_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            # parameters for decoder
            "decoder_dim": 512,
            # parameters for joiner
            "joiner_dim": 512,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            # "env_info": get_env_info(),
            "dynamic_chunk_training": False,
            "short_chunk_size": 25,
            "num_left_chunks": 4,
            "causal_convolution": False,
            "context_size": 2,
        }
    )


def build_conformer_transducer_model(sp, params):
    if isinstance(sp, Lexicon):
        params.blank_id = sp.token_table["<blk>"]
        params.vocab_size = max(sp.tokens) + 1
        logger.info(f"vocab size: {params.vocab_size}")
    else:
        params.blank_id = sp.piece_to_id("<blk>")
        params.vocab_size = sp.get_piece_size()

    encoder = Conformer(
        num_features=params.feature_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.encoder_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        dynamic_chunk_training=params.dynamic_chunk_training,
        short_chunk_size=params.short_chunk_size,
        num_left_chunks=params.num_left_chunks,
        causal=params.causal_convolution,
    )
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    joiner = Joiner(
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    model = Transducer(
        encoder,
        decoder,
        joiner,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model
