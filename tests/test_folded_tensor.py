import pytest
import torch

from foldedtensor import FoldedTensor, as_folded_tensor


def test_as_folded_tensor_from_nested_list():
    ft = as_folded_tensor(
        [
            [[1], [], [], [], [2, 3]],
            [[4, 3]],
        ],
        data_dims=("samples", "lines", "words"),
        full_names=("samples", "lines", "words"),
        dtype=torch.long,
        device=torch.device("cpu"),
    )
    assert ft.data.shape == (2, 5, 2)
    assert ft.lengths == [[2], [5, 1], [1, 0, 0, 0, 2, 2]]
    assert ft.data_dims == (0, 1, 2)
    assert ft.full_names == ("samples", "lines", "words")
    assert (
        ft.data
        == torch.tensor(
            [
                [[1, 0], [0, 0], [0, 0], [0, 0], [2, 3]],
                [[4, 3], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        )
    ).all()
    assert (
        ft.mask
        == torch.tensor(
            [
                [[1, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ).bool()
    ).all()
    assert ft.mask.dtype == torch.bool


def test_embedding_from_nested_list():
    ft = as_folded_tensor(
        [
            [[[1, 1, 1]], [], [], [], [[2, 2, 2], [3, 3, 3]]],
            [[[4, 4, 4], [3, 3, 3]]],
        ],
        data_dims=("samples", "lines", "words"),
        full_names=("samples", "lines", "words"),
        dtype=torch.long,
    )
    assert ft.data.shape == (2, 5, 2, 3)
    assert ft.lengths == [[2], [5, 1], [1, 0, 0, 0, 2, 2]]
    assert ft.data_dims == (0, 1, 2)
    assert ft.full_names == ("samples", "lines", "words")
    assert (
        ft.data[..., 0]
        == torch.tensor(
            [
                [[1, 0], [0, 0], [0, 0], [0, 0], [2, 3]],
                [[4, 3], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        )
    ).all()
    assert (
        ft.mask
        == torch.tensor(
            [
                [[1, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ).bool()
    ).all()
    assert ft.mask.dtype == torch.bool


def test_as_folded_tensor_from_tensor():
    ft = as_folded_tensor(
        torch.tensor(
            [
                [[1, 0], [0, 0], [0, 0], [0, 0], [2, 3]],
                [[4, 3], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ),
        data_dims=("samples", "lines", "words"),
        full_names=("samples", "lines", "words"),
        lengths=[[2], [5, 1], [1, 0, 0, 0, 2, 2]],
    )
    assert ft.data.shape == (2, 5, 2)
    assert ft.lengths == [[2], [5, 1], [1, 0, 0, 0, 2, 2]]
    assert ft.data_dims == (0, 1, 2)
    assert ft.full_names == ("samples", "lines", "words")
    assert (
        ft.data
        == torch.tensor(
            [
                [[1, 0], [0, 0], [0, 0], [0, 0], [2, 3]],
                [[4, 3], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        )
    ).all()


def test_as_folded_tensor_error():
    with pytest.raises(ValueError) as excinfo:
        as_folded_tensor(
            [
                [[1], [], [], [], [2, 3]],
                [[4, 3]],
            ],
            data_dims=(
                "samples",
                "lines",
            ),
            full_names=("samples", "lines", "words"),
            dtype=torch.long,
        )

    assert "The last dimension" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        as_folded_tensor(
            torch.tensor(
                [
                    [[1, 0], [0, 0], [0, 0], [0, 0], [2, 3]],
                    [[4, 3], [0, 0], [0, 0], [0, 0], [0, 0]],
                ]
            ),
            data_dims=("samples", "lines", "words"),
            full_names=("samples", "lines", "words"),
        )

    assert "as_folded_tensor expects:" in str(excinfo.value)


@pytest.fixture
def ft():
    return as_folded_tensor(
        [
            [[1], [], [], [], [2, 3]],
            [[4, 3]],
        ],
        data_dims=("samples", "lines", "words"),
        full_names=("samples", "lines", "words"),
        dtype=torch.long,
    )


def test_refold_samples(ft):
    ft2 = ft.refold("samples", "words")
    assert ft2.data.shape == (2, 3)
    assert ft2.lengths == [[2], [5, 1], [1, 0, 0, 0, 2, 2]]
    assert ft2.data_dims == (0, 2)
    assert ft2.full_names == ("samples", "lines", "words")
    assert (
        ft2.data
        == torch.tensor(
            [
                [1, 2, 3],
                [4, 3, 0],
            ]
        )
    ).all()
    assert (
        ft2.mask
        == torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ]
        ).bool()
    ).all()
    assert ft2.mask.dtype == torch.bool


def test_refold_tuple_param(ft):
    ft2 = ft.refold(("samples", "words"))
    assert (
        ft2.data
        == torch.tensor(
            [
                [1, 2, 3],
                [4, 3, 0],
            ]
        )
    ).all()


def test_refold_noop(ft):
    ft2 = ft.refold("samples", "lines", "words")
    assert ft2 is ft


def test_refold_lines(ft):
    ft2 = ft.refold("lines", "words")
    assert ft2.data.shape == (6, 2)
    assert ft2.lengths == [[2], [5, 1], [1, 0, 0, 0, 2, 2]]
    assert ft2.data_dims == (1, 2)
    assert ft2.full_names == ("samples", "lines", "words")
    assert (
        ft2.data
        == torch.tensor(
            [
                [1, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [2, 3],
                [4, 3],
            ]
        )
    ).all()


def test_embedding(ft):
    embedder = torch.nn.Embedding(10, 16)
    embedding = embedder(ft.refold("words"))
    assert isinstance(embedding, FoldedTensor)
    assert embedding.data.shape == (5, 16)
    refolded = embedding.refold("samples", "words")
    assert refolded.data.shape == (2, 3, 16)


def test_embedding_backward(ft):
    embedder = torch.nn.Embedding(6, 7)
    embedding = embedder(ft.refold("words")).refold("samples", "words")
    embedding.sum().backward()
    assert embedder.weight.grad.shape == (6, 7)
    assert (
        embedder.weight.grad.data
        == torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()


def test_to(ft):
    # this test doesn't do much since we don't have access to non-cpu devices in the CI
    assert ft.to("cpu").indexer.device == torch.device("cpu")


def test_with_data(ft):
    new_ft = ft.with_data(ft.as_tensor() + 1)
    assert (
        new_ft.refold("samples", "words") == torch.tensor([[2, 3, 4], [5, 4, 0]])
    ).all()


def test_list_args(ft):
    embedder = torch.nn.Embedding(10, 16)
    embedding = embedder(ft.refold("words"))
    cat_ft = torch.cat([embedding, embedding], dim=-1)
    assert isinstance(cat_ft, FoldedTensor)
    assert cat_ft.data.shape == (5, 32)
    refolded = cat_ft.refold("samples", "words")
    assert refolded.data.shape == (2, 3, 32)


def test_too_deep():
    with pytest.raises(ValueError) as excinfo:
        as_folded_tensor(
            [
                [0, 1, 2],
                [3, 4],
            ],
            full_names=("sample", "line", "token"),
            data_dims=("sample", "line", "token"),
            dtype=torch.long,
        )
    assert "nesting" in str(excinfo.value)


def test_pad_embedding():
    ft = as_folded_tensor(
        [
            [0, 1, 2],
            [3, 4],
        ],
        full_names=("token",),
        data_dims=("token",),
        dtype=torch.long,
    )
    assert (
        ft.data
        == torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 0],
            ]
        )
    ).all()


def test_empty_args():
    ft = as_folded_tensor(
        [
            [0, 1, 2],
            [3, 4],
        ],
    )
    assert (
        ft.data
        == torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 0],
            ]
        )
    ).all()
    assert ft.data.dtype == torch.int64
    assert (
        ft.mask
        == torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ]
        ).bool()
    ).all()


def test_no_data_dims():
    ft = as_folded_tensor(
        [
            [0, 1, 2],
            [3, 4],
        ],
        full_names=("token",),
        dtype=torch.long,
    )
    assert (
        ft.data
        == torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 0],
            ]
        )
    ).all()


def test_as_tensor(ft):
    tensor = ft.as_tensor()
    assert type(tensor) == torch.Tensor
    assert tensor.shape == (2, 5, 2)
    assert tensor.storage().data_ptr() == ft.storage().data_ptr()


def test_clone(ft):
    assert isinstance(ft.mask, torch.Tensor)
    cloned = ft.clone()
    assert cloned.storage().data_ptr() != ft.storage().data_ptr()
    assert cloned.indexer.storage().data_ptr() != ft.indexer.storage().data_ptr()


def test_share_memory(ft):
    assert isinstance(ft.mask, torch.Tensor)
    cloned = ft.share_memory_()
    assert cloned.is_shared()
    assert cloned.indexer.is_shared()
    assert cloned.mask.is_shared()


def test_empty_sequence():
    ft = as_folded_tensor(
        [
            [[], [], []],
            [[], []],
        ],
        dtype=torch.float,
    )
    assert ft.shape == (2, 3, 0)


def test_imbalanced_sequence_1():
    with pytest.raises(ValueError) as e:
        as_folded_tensor(
            [
                3,
                [0, 1, 2],
            ],
            dtype=torch.float,
        )

    assert "setting an array element with a sequence." in str(e.value)


def test_imbalanced_sequence_2():
    with pytest.raises(TypeError) as e:
        as_folded_tensor(
            [
                [0, 1, 2],
                3,
            ],
            dtype=torch.float,
        )

    assert "'int' object is not iterable" in str(e.value)


def test_max():
    ft = as_folded_tensor(
        [
            [0, 1, 2],
            [3, 4],
        ],
        dtype=torch.float,
    )
    values, indices = ft.max(-1)
    assert (values == torch.tensor([2, 4])).all()
    assert (indices == torch.tensor([2, 1])).all()


def test_hashable_lengths():
    tensor = as_folded_tensor(
        [
            [0, 1, 2],
            [3, 4],
        ],
        dtype=torch.long,
    )
    embedding = torch.nn.Embedding(10, 16)
    assert tensor.lengths is embedding(tensor).lengths
    assert hash(tensor.lengths) is not None
    assert hash(tensor.lengths) == hash(embedding(tensor).lengths)


def test_missing_dims():
    tensor = as_folded_tensor(
        [
            [0, 1, 2],
            [3, 4],
        ],
        full_names=("sample", "token"),
        dtype=torch.long,
    )
    with pytest.raises(ValueError) as e:
        tensor.refold("line", "token")

    assert "line" in str(e.value)
