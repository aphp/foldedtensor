import numpy as np
import torch

import foldedtensor as ft


def build_tensor():
    # Two samples total. First sample mirrors the example in the prompt
    # and totals 14 tokens (contexts: 5 and 9). Second sample has one
    # small context to ensure strides are unchanged for the (context, token) view.
    data = [
        [
            [
                [0, 2, 3],
                [10],
                [4],
            ],
            [
                [0, 1, 2],
                [2, 3],
                [10, 11],
                [100, 101],
            ],
        ],
        [
            [
                [7],
                [8, 9],
            ],
        ],
    ]
    return ft.as_folded_tensor(data, full_names=("sample", "context", "word", "token"))


def build_tensor_single_sample():
    # Single sample as in the prompt example
    data = [
        [
            [
                [0, 2, 3],
                [10],
                [4],
            ],
            [
                [0, 1, 2],
                [2, 3],
                [10, 11],
                [100, 101],
            ],
        ],
    ]
    return ft.as_folded_tensor(data, full_names=("sample", "context", "word", "token"))


def test_map_indices_flat_unpadded_tokens_by_token():
    t = build_tensor()

    assert t.refold("token").lengths.map_indices(
        indices=([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],),
        indice_dims=("token",),
    ) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_map_indices_flat_unpadded_tokens_by_word():
    t = build_tensor()

    assert t.refold("token").lengths.map_indices(
        indices=([0, 1, 2, 3, 4, 5, 6, 7],),
        indice_dims=("word",),
    ) == [0, 3, 4, 5, 8, 10, 12, 14]


def test_map_indices_flat_unpadded_tokens_by_context():
    t = build_tensor()

    assert t.refold("token").lengths.map_indices(
        indices=([0, 1, 2],),
        indice_dims=("context",),
    ) == [0, 5, 14]


def test_map_indices_flat_unpadded_tokens_by_sample():
    t = build_tensor()

    assert t.refold("token").lengths.map_indices(
        indices=([0, 1],),
        indice_dims=("sample",),
    ) == [0, 14]


def test_map_indices_subset_words():
    t = build_tensor()

    assert t.refold("token").lengths.map_indices(
        indices=([0, 1, 2, 4, 6],),
        indice_dims=("word",),
    ) == [0, 3, 4, 8, 12]


def test_map_indices_context_word_to_padded_context_token():
    t = build_tensor()

    assert t.refold("context", "token").lengths.map_indices(
        indices=([0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]),
        indice_dims=("context", "word"),
    ) == [0, 3, 4, 5, 9, 12, 14, 16]


def test_make_indices_ranges_flat_tokens():
    t = build_tensor_single_sample()

    indices, offsets, spans = t.refold("token").lengths.make_indices_ranges(
        begins=(torch.as_tensor([0, 0, 1]), torch.as_tensor([0, 1, 2])),
        ends=(torch.as_tensor([0, 1, 1]), torch.as_tensor([1, 3, 4])),
        indice_dims=("context", "word"),
        return_tensors=False,
    )

    assert indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 11, 12, 13]
    assert offsets == [0, 3, 12]
    assert spans == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]


def test_make_indices_ranges_one_dim_token():
    t = build_tensor_single_sample()
    indices, offsets, spans = t.refold("token").lengths.make_indices_ranges(
        begins=(torch.tensor([0, 3, 12]),),
        ends=(torch.tensor([3, 12, 14]),),
        indice_dims=("token",),
        return_tensors=False,
    )
    assert indices == list(range(0, 3)) + list(range(3, 12)) + list(range(12, 14))
    assert offsets == [0, 3, 12]
    assert spans == [0] * 3 + [1] * 9 + [2] * 2


def test_make_indices_ranges_one_dim_word():
    t = build_tensor_single_sample()
    indices, offsets, spans = t.refold("token").lengths.make_indices_ranges(
        begins=(torch.tensor([0, 1, 3]),),
        ends=(torch.tensor([1, 3, 7]),),
        indice_dims=("word",),
        return_tensors=False,
    )
    assert indices == list(range(0, 3)) + list(range(3, 5)) + list(range(5, 14))
    assert offsets == [0, 3, 5]
    assert spans == [0] * 3 + [1] * 2 + [2] * 9


def test_word_span_mean_pooler_with_embedding_bag_flat_indices():
    t = build_tensor().refold("context", "token")
    # 0 -> 2: [[0, 2, 3], [10]]
    # 5 -> 7: [[10, 11], [100, 101]]
    indices, offsets, spans = t.lengths.make_indices_ranges(
        begins=(torch.tensor([0, 5]),),
        ends=(torch.tensor([2, 7]),),
        indice_dims=("word",),
    )
    embeds = t.unsqueeze(-1).expand(-1, -1, 2).float()
    res = torch.nn.functional.embedding_bag(
        input=indices,
        weight=embeds.view(-1, 2),
        offsets=offsets,
        mode="mean",
    )
    assert res.tolist() == [[3.75, 3.75], [55.5, 55.5]]


def test_word_span_mean_pooler_with_embedding_bag_multidim_indices():
    t = build_tensor().refold("context", "token")
    # 0 -> 2: [[0, 2, 3], [10]]
    # 5 -> 7: [[10, 11], [100, 101]]
    indices, offsets, spans = t.lengths.make_indices_ranges(
        begins=(
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
        ),
        ends=(
            torch.tensor([0, 1]),
            torch.tensor([2, 4]),
        ),
        indice_dims=(
            "context",
            "word",
        ),
    )
    embeds = t.unsqueeze(-1).expand(-1, -1, 2).float()
    res = torch.nn.functional.embedding_bag(
        input=indices,
        weight=embeds.view(-1, 2),
        offsets=offsets,
        mode="mean",
    )
    assert res.tolist() == [[3.75, 3.75], [55.5, 55.5]]


def test_map_indices_format_torch_multidimensional():
    t = build_tensor()

    assert torch.allclose(
        t.lengths.map_indices(
            indices=(
                torch.as_tensor([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]),
            ),
            indice_dims=("token",),
        ),
        torch.tensor([[0, 1, 2, 3, 6, 12, 13], [14, 15, 16, 18, 19, 21, 22]]),
    )


def test_map_indices_format_numpy_multidimensional():
    t = build_tensor()

    assert np.allclose(
        t.lengths.map_indices(
            indices=(np.asarray([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]),),
            indice_dims=("token",),
        ),
        np.asarray([[0, 1, 2, 3, 6, 12, 13], [14, 15, 16, 18, 19, 21, 22]]),
    )


def test_make_indices_ranges_format_torch_multidimensional():
    t = build_tensor_single_sample()

    indices, offsets, spans = t.lengths.make_indices_ranges(
        begins=(torch.as_tensor([[0, 0, 1]]), torch.as_tensor([[0, 1, 2]])),
        ends=(torch.as_tensor([[0, 1, 1]]), torch.as_tensor([[1, 3, 4]])),
        indice_dims=("context", "word"),
    )

    assert isinstance(indices, torch.Tensor)
    assert isinstance(offsets, torch.Tensor)
    assert isinstance(spans, torch.Tensor)
    assert offsets.shape == (1, 3)


def test_make_indices_ranges_format_numpy_multidimensional():
    t = build_tensor_single_sample()

    indices, offsets, spans = t.lengths.make_indices_ranges(
        begins=(np.asarray([[0, 0, 1]]), np.asarray([[0, 1, 2]])),
        ends=(np.asarray([[0, 1, 1]]), np.asarray([[1, 3, 4]])),
        indice_dims=("context", "word"),
    )
    assert isinstance(indices, np.ndarray)
    assert isinstance(offsets, np.ndarray)
    assert isinstance(spans, np.ndarray)
    assert offsets.shape == (1, 3)
