import numpy as np
import pytest
import torch

import foldedtensor as ft


def build_tensor():
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


@pytest.mark.parametrize("data_dims", [(3,), (1, 3), (0, 1, 2, 3)])
@pytest.mark.parametrize(
    "dims,begins,ends,expected,offsets",
    [
        (("token",), ([0, 3, 14],), ([3, 14, 17],), list(range(17)), [0, 3, 14]),
        (("word",), ([0, 1, 3],), ([1, 3, 9],), list(range(17)), [0, 3, 5]),
        (("context",), ([0, 1, 2],), ([1, 2, 3],), list(range(17)), [0, 5, 14]),
        (("sample",), ([0, 1],), ([1, 2],), list(range(17)), [0, 14]),
        (
            ("context", "word"),
            ([0, 0, 1], [0, 1, 2]),
            ([0, 1, 1], [1, 3, 4]),
            list(range(12)) + list(range(10, 14)),
            [0, 3, 12],
        ),
        (
            ("sample", "word"),
            ([0, 1], [2, 1]),
            ([0, 1], [4, 2]),
            [4, 5, 6, 7, 15, 16],
            [0, 4],
        ),
    ],
)
def test_ranges(data_dims, dims, begins, ends, expected, offsets):
    tensor = build_tensor().refold(*data_dims)
    indices, actual_offsets, owners = tensor.lengths.make_indices_ranges(
        begins=begins,
        ends=ends,
        indice_dims=dims,
    )
    assert indices == tensor.indexer[expected].tolist()
    assert actual_offsets == offsets
    assert owners == [
        i
        for i, (a, b) in enumerate(zip(offsets, offsets[1:] + [len(expected)]))
        for _ in range(a, b)
    ]


@pytest.mark.parametrize("array", [torch.as_tensor, np.asarray])
def test_ranges_broadcast_and_output_type(array):
    tensor = build_tensor()
    indices, offsets, owners = tensor.lengths.make_indices_ranges(
        begins=(array([[0], [1]]), array([0, 1])),
        ends=(array([[0], [1]]), array([1, 3])),
        indice_dims=("context", "word"),
    )
    assert all(
        type(x) is type(array([])) for x in (indices, offsets, owners)  # noqa: E721
    )
    assert offsets.tolist() == [[0, 3], [5, 8]]
    assert (
        indices.tolist()
        == tensor.indexer[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].tolist()
    )


@pytest.mark.parametrize("dims", [(2,), (0, 2), (0, 1, 2)])
def test_ranges_and_refolding_with_empty_contexts(dims):
    # Empty parents keep their rows when the pooler gathers padded wordpieces
    layout = ft.FoldedTensorLayout(
        [[3], [0, 4, 2], [2, 0, 1, 3, 1, 2]],
        data_dims=(2,),
        full_names=("context", "word", "piece"),
    )
    values = torch.arange(9, dtype=torch.float32, requires_grad=True)
    tensor = ft.as_folded_tensor(values, lengths=layout).refold(*dims)
    nested = ft.as_folded_tensor(
        [[], [[0.0, 1.0], [], [2.0], [3.0, 4.0, 5.0]], [[6.0], [7.0, 8.0]]],
        data_dims=dims,
    )
    assert torch.equal(tensor.as_tensor(), nested.as_tensor())
    indices, offsets, owners = tensor.lengths.make_indices_ranges(
        begins=([0, 1, 1, 1, 1, 2], [0, 0, 1, 1, 2, 0]),
        ends=([0, 1, 1, 1, 1, 2], [0, 4, 1, 4, 3, 2]),
        indice_dims=("context", "word"),
    )
    expected = list(range(6)) + list(range(2, 6)) + [2] + list(range(6, 9))
    actual = tensor.as_tensor().reshape(-1)[indices]
    assert actual.tolist() == expected
    assert offsets == [0, 0, 6, 6, 10, 11]
    assert owners == [1] * 6 + [3] * 4 + [4] + [5] * 3
    actual.sum().backward()
    assert values.grad.tolist() == np.bincount(expected, minlength=9).tolist()


def test_ranges_empty_and_scalar():
    tensor = ft.as_folded_tensor([[], []], full_names=("sample", "word"))
    assert tensor.refold("word").refold("sample", "word").shape == (2, 0)
    assert tensor.lengths.make_indices_ranges(
        begins=([0, 1], [0]),
        ends=([0, 1], [0]),
        indice_dims=("sample", "word"),
    ) == ([], [0, 0], [])
    assert tensor.lengths.make_indices_ranges(
        begins=([],),
        ends=([],),
        indice_dims=("word",),
    ) == ([], [], [])
    assert build_tensor().lengths.make_indices_ranges(
        begins=(0,),
        ends=(1,),
        indice_dims=("token",),
    ) == ([0], 0, [0])


@pytest.mark.parametrize(
    "begins,ends,dims,error",
    [
        (([-1],), ([1],), ("word",), IndexError),
        (([0], [4]), ([0], [4]), ("context", "word"), IndexError),
        (([0], [0]), ([3], [0]), ("context", "word"), IndexError),
        (([2],), ([1],), ("word",), ValueError),
        (([0], [0]), ([1], [1]), ("word", "word"), ValueError),
    ],
)
def test_invalid_ranges(begins, ends, dims, error):
    with pytest.raises(error):
        build_tensor().lengths.make_indices_ranges(
            begins=begins, ends=ends, indice_dims=dims
        )
