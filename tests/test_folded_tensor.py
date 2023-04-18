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

    assert "as_folded_tensor expects either:" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        as_folded_tensor(
            [
                [[1], [], [], [], [2, 3]],
                [[4, 3]],
            ],
            data_dims=("samples", "lines", "words"),
            full_names=("samples", "lines", "words"),
            lengths=[[1, 2, 3]],
        )

    assert "as_folded_tensor expects either:" in str(excinfo.value)


@pytest.fixture
def ft():
    return as_folded_tensor(
        [
            [[1], [], [], [], [2, 3]],
            [[4, 3]],
        ],
        data_dims=("samples", "lines", "words"),
        full_names=("samples", "lines", "words"),
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
