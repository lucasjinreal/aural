from aural.utils.util import AttributeDict
from .transducer import Transducer
from ..encoders.rnn import RNNEncoder, RNN
from ..decoders.decoder import Decoder
from ..post.joiner import Joiner


def build_lstm_transducer_model(sp):
    params = AttributeDict(
        {
            "num_encoder_layers": 12,
            "encoder_dim": 512,
            "rnn_hidden_size": 1024,
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
            "dim_feedforward": 2048,
            # parameters for decoder
            "decoder_dim": 512,
            # parameters for joiner
            "joiner_dim": 512,
            # True to generate a model that can be exported via PNNX
            "is_pnnx": False,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            # "env_info": get_env_info(),
            "context_size": 2,
        }
    )

    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    encoder = RNN(
        num_features=params.feature_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.encoder_dim,
        rnn_hidden_size=params.rnn_hidden_size,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        # aux_layer_period=params.aux_layer_period,
        is_pnnx=params.is_pnnx,
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
