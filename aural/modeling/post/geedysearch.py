import math
from typing import List
import torch

from aural.modeling.post.beamsearch import greedy_search
from torch.nn.utils.rnn import pad_sequence
from alfred import device


def greedy_search_single_batch(
    model, encoder_out: torch.Tensor, max_sym_per_frame: int=1
) -> List[int]:
    """Greedy search for a single utterance.
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      max_sym_per_frame:
        Maximum number of symbols per frame. If it is set to 0, the WER
        would be 100%.
    Returns:
      Return the decoded result.
    """
    assert encoder_out.ndim == 3
    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    unk_id = getattr(model, "unk_id", blank_id)

    device = next(model.parameters()).device
    decoder_input = torch.tensor(
        [blank_id] * context_size, device=device, dtype=torch.int64
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out.size(1)
    t = 0
    hyp = [blank_id] * context_size

    # Maximum symbols per utterance.
    max_sym_per_utt = 1000
    # symbols per frame
    sym_per_frame = 0
    # symbols per utterance decoded so far
    sym_per_utt = 0

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits is (1, 1, 1, vocab_size)
        y = logits.argmax().item()
        if y not in (blank_id, unk_id):
            hyp.append(y)
            decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(
                1, context_size
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1
    hyp = hyp[context_size:]  # remove blanks
    return hyp


def greedy_search_batch(
    model,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
) -> List[List[int]]:
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    device = next(model.parameters()).device
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[blank_id] * context_size for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end
        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits'shape (batch_size, 1, 1, vocab_size)
        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
    return ans

LOG_EPS = math.log(1e-10)


class GreedySearchOffline:
    def __init__(self):
        pass

    @torch.no_grad()
    def process(
        self,
        model,
        features: List[torch.Tensor],
    ) -> List[List[int]]:
        """
        Args:
          model:
            RNN-T model decoder model
          features:
            A list of 2-D tensors. Each entry is of shape
            (num_frames, feature_dim).
        Returns:
          Return a list-of-list containing the decoding token IDs.
        """
        features_length = torch.tensor(
            [f.size(0) for f in features],
            dtype=torch.int64,
        )
        features = pad_sequence(
            features,
            batch_first=True,
            padding_value=LOG_EPS,
        )

        features = features.to(device)
        features_length = features_length.to(device)

        if device.type == "cpu":
            encoder_out, encoder_out_length = model.encoder(
               features,
                features_length,
            )
        elif device.type == "cuda":
            with torch.cuda.amp.autocast():
                encoder_out, encoder_out_length = model.encoder(
                    features,
                    features_length,
                )
        else:
            raise NotImplementedError("Device not supported")

        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_length.cpu(),
        )
        return hyp_tokens