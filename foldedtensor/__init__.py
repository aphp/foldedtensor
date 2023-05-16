from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.autograd import Function

numpy_to_torch_dtype_dict = {
    torch.bool: bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}

from . import _C

__version__ = "0.1.0"

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

        np_old_indexer, np_new_indexer, shape, np_mask = _C.make_refolding_indexers(
            self.lengths,
            self.data.shape,
            self.data_dims,
            dims,
        )
        shape_suffix = self.data.shape[len(self.data_dims) :]
        flat_data = self.data.view(-1, *shape_suffix)
        refolded_data = torch.zeros(
            shape, dtype=self.data.dtype, device=self.data.device
        )
        refolded_data = refolded_data.view(-1, *shape_suffix)
        refolded_data.index_put_(
            indices=(torch.from_numpy(np_new_indexer),),
            values=flat_data.index_select(0, torch.from_numpy(np_old_indexer)),
        )
        return FoldedTensor(
            refolded_data.view(shape),
            self.lengths,
            dims,
            self.full_names,
            mask=torch.from_numpy(np_mask),
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Perform inverse refolding, i.e., dims = old data_dims
        np_old_indexer, np_new_indexer, shape, np_mask = _C.make_refolding_indexers(
            ctx.lengths,
            grad_output.shape,
            ctx.dims,
            ctx.old_data_dims,
        )
        # new_data_flat.index_put_({new_indexer}, old_data_flat.index_select(0, old_indexer));
        shape_suffix = grad_output.shape[len(ctx.dims) :]
        flat_data = grad_output.view(-1, *shape_suffix)
        refolded_data = torch.zeros(
            shape, dtype=grad_output.dtype, device=grad_output.device
        )
        refolded_data = refolded_data.view(-1, *shape_suffix)
        refolded_data.index_put_(
            indices=(torch.from_numpy(np_new_indexer),),
            values=flat_data.index_select(0, torch.from_numpy(np_old_indexer)),
        )
        return refolded_data.view(shape), None


def as_folded_tensor(
    data: Union[torch.Tensor, Sequence],
    data_dims: Sequence[Union[int, str]],
    full_names: Sequence[str],
    dtype: Optional[torch.dtype] = None,
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
    dtype: Optional[torch.dtype]
        The dtype of the output tensor
    lengths: Optional[List[List[int]]]
        The lengths of the variable dimensions. If `data` is a tensor, this argument
        must be provided. If `data` is a sequence, this argument must be `None`.
    """
    data_dims = tuple(
        dim if isinstance(dim, int) else full_names.index(dim) for dim in data_dims
    )
    dtype = numpy_to_torch_dtype_dict.get(dtype, dtype)
    if (data_dims[-1] + 1) != len(full_names):
        raise ValueError(
            "The last dimension of `data_dims` must be the last variable dimension."
        )
    if isinstance(data, torch.Tensor) and lengths is not None:
        return FoldedTensor(data, lengths, data_dims, full_names)
    elif isinstance(data, Sequence) and lengths is None:
        assert dtype is not None, "dtype must be provided when `data` is a sequence"
        padded, mask, lengths = _C.nested_py_list_to_padded_array(
            data, data_dims, np.dtype(dtype)
        )
        return FoldedTensor(
            torch.from_numpy(padded),
            lengths,
            data_dims,
            full_names,
            torch.from_numpy(mask),
        )
    else:
        raise ValueError(
            "as_folded_tensor expects either:\n"
            "- a `data` tensor with a `lengths` argument\n"
            "- a `data` (optionally nested) sequence with no `lengths` argument"
        )


# noinspection PyUnresolvedReferences,PyInitNewSignature
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
        if isinstance(result, torch.Tensor) and ft is not None:
            result = FoldedTensor(
                result,
                ft.lengths,
                ft.data_dims,
                ft.full_names,
            )

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
