from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.autograd import Function

from . import _C

try:
    from .version import __version__  # noqa: F401
except ImportError:
    __version__ = None


# noinspection PyMethodOverriding
class Refold(Function):
    @staticmethod
    def forward(
        ctx,
        self: "FoldedTensor",
        dims: Tuple[int],
    ) -> "FoldedTensor":
        ctx.set_materialize_grads(False)
        ctx.lengths = self.lengths
        ctx.old_data_dims = self.data_dims
        ctx.dims = dims

        refolded_data, mask = _C.refold(
            self.data,
            self.lengths,
            self.data_dims,
            dims,
            True,
        )
        return FoldedTensor(
            refolded_data,
            self.lengths,
            dims,
            self.full_names,
            mask=mask,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Perform inverse refolding, i.e., dims = old data_dims
        grad_input = _C.refold(
            grad_output,
            ctx.lengths,
            ctx.dims,
            ctx.old_data_dims,
            False,
        )[0]
        return grad_input, None


def as_folded_tensor(
    data: Union[torch.Tensor, Sequence],
    data_dims: Sequence[Union[int, str]],
    full_names: Sequence[str],
    lengths: Optional[List[List[int]]] = None,
):
    """
    Converts a tensor or nested sequence into a FoldedTensor.

    Parameters
    ----------
    data: Union[torch.Tensor, Sequence]
        The data to convert into a FoldedTensor.
    data_dims: Sequence[Union[int, str]]
        The flattened dimensions of the data tensor. The last dim must be the last
        variable dimension.
    full_names: List[str]
        The names of the variable dimensions.
    lengths: Optional[List[List[int]]]
        The lengths of the variable dimensions. If `data` is a tensor, this argument
        must be provided. If `data` is a sequence, this argument must be `None`.
    """
    data_dims = tuple(
        dim if isinstance(dim, int) else full_names.index(dim) for dim in data_dims
    )
    if (data_dims[-1] + 1) != len(full_names):
        raise ValueError(
            "The last dimension of `data_dims` must be the last variable dimension."
        )
    if isinstance(data, torch.Tensor) and lengths is not None:
        return FoldedTensor(data, lengths, data_dims, full_names)
    elif isinstance(data, Sequence) and lengths is None:
        padded, mask, lengths = _C.nested_py_list_to_padded_tensor(data, data_dims)
        return FoldedTensor(padded, lengths, data_dims, full_names, mask)
    else:
        raise ValueError(
            "as_folded_tensor expects either:\n"
            "- a `data` tensor with a `lengths` argument\n"
            "- a `data` (optionally nested) sequence with no `lengths` argument"
        )


# noinspection PyUnresolvedReferences
class FoldedTensor(torch.Tensor):
    """
    A FoldedTensor is an extension of Pytorch Tensors that provides operations for
    efficiently handling tensors containing deeply nested sequences variable sizes.
    It enables the flattening/unflattening (or unfolding/folding) of data dimensions
    based on a inner structure of sequence lengths. This library is particularly useful
    when working with data that can be split in different ways and enables you to
    avoid choosing a fixed representation.

    Parameters
    ----------
    data: torch.Tensor
        The data tensor.
    lengths: List[List[int]]
        The lengths of the sequences of variable size, one list for each dimension.
    data_dims: Sequence[int]
        The flattened dimensions of the data tensor. The last dim must be the last
        variable dimension.
    full_names: Sequence[str]
        The names of the variable dimensions.
    mask: Optional[torch.Tensor]
        A mask tensor that indicates which elements of the data tensor are not padded.
    """

    def __new__(
        cls,
        data: torch.Tensor,
        lengths: List[List[int]],
        data_dims: Sequence[int],
        full_names: Sequence[str],
        mask: Optional[torch.Tensor] = None,
    ):
        data_dims = data_dims
        full_names = full_names
        instance = data.as_subclass(cls)
        instance.lengths = lengths
        instance.data_dims = data_dims
        instance.full_names = full_names
        instance.mask = mask
        return instance

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        new_args = []
        ft = None
        for arg in args:
            if isinstance(arg, FoldedTensor):
                assert (
                    ft is None or ft.data_dims == arg.data_dims
                ), "Cannot perform operation on FoldedTensors with different structure"
                ft = arg
                new_args.append(arg)
            else:
                new_args.append(arg)

        result = super().__torch_function__(func, types, args, kwargs)
        if isinstance(result, torch.Tensor):
            result = FoldedTensor(result, ft.lengths, ft.data_dims, ft.full_names)

        return result

    def refold(self, *dims: Union[Sequence[Union[int, str]], int, str]):
        if not isinstance(dims[0], (int, str)):
            assert len(dims) == 1, (
                "Expected the first only argument to be a "
                "sequence or each arguments to be ints or strings"
            )
            dims = dims[0]
        dims = tuple(
            dim if isinstance(dim, int) else self.full_names.index(dim) for dim in dims
        )

        if dims == self.data_dims:
            return self

        return Refold.apply(self, dims)
