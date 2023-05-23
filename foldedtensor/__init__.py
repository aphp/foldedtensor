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

__version__ = "0.2.1.post0"


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
        ctx.new_data_dims = dims
        ctx.indexer = self.indexer

        device = self.data.device

        np_new_indexer, shape_prefix = _C.make_refolding_indexer(
            self.lengths,
            dims,
        )
        data = self.as_tensor()
        indexer = torch.from_numpy(np_new_indexer).to(device)
        shape_suffix = data.shape[len(self.data_dims) :]
        refolded_data = torch.zeros(
            (*shape_prefix, *shape_suffix), dtype=data.dtype, device=device
        )
        refolded_data.view(-1, *shape_suffix)[indexer] = data.view(
            -1, *shape_suffix
        ).index_select(0, self.indexer)
        return FoldedTensor(
            data=refolded_data,
            lengths=self.lengths,
            data_dims=dims,
            full_names=self.full_names,
            indexer=indexer,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Perform inverse refolding, i.e., dims = old data_dims
        device = grad_output.device
        # full_names = grad_output.full_names
        np_new_indexer, shape_prefix = _C.make_refolding_indexer(
            ctx.lengths,
            ctx.old_data_dims,
        )
        indexer = torch.from_numpy(np_new_indexer).to(device)
        # new_data_flat.index_put_({new_indexer}, old_data_flat.index_select(0, old_indexer));
        shape_suffix = grad_output.shape[len(ctx.new_data_dims) :]
        grad_input = torch.zeros(
            (*shape_prefix, *shape_suffix), dtype=grad_output.dtype, device=device
        )
        index_select = grad_output.reshape(-1, *shape_suffix).index_select(
            0, ctx.indexer
        )
        grad_input.view(-1, *shape_suffix)[indexer] = index_select
        return grad_input, None
        # return FoldedTensor(
        #     data=refolded_data,
        #     lengths=ctx.lengths,
        #     data_dims=ctx.old_data_dims,
        #     full_names=full_names,
        #     indexer=indexer,
        # )


def as_folded_tensor(
    data: Sequence,
    data_dims: Sequence[Union[int, str]],
    full_names: Sequence[str],
    dtype: Optional[torch.dtype] = None,
    lengths: Optional[List[List[int]]] = None,
    device: Optional[torch.device] = None,
):
    """
    Converts a tensor or nested sequence into a FoldedTensor.

    Parameters
    ----------
    data: Sequence
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
    device: Optional[torch.device]
        The device of the output tensor
    """
    data_dims = tuple(
        dim if isinstance(dim, int) else full_names.index(dim) for dim in data_dims
    )
    if (data_dims[-1] + 1) != len(full_names):
        raise ValueError(
            "The last dimension of `data_dims` must be the last variable dimension."
        )
    if isinstance(data, Sequence):
        if dtype is None:
            raise ValueError("dtype must be provided when `data` is a sequence")
        dtype = numpy_to_torch_dtype_dict.get(dtype, dtype)
        padded, indexer, lengths = _C.nested_py_list_to_padded_array(
            data,
            data_dims,
            np.dtype(dtype),
        )
        indexer = torch.from_numpy(indexer)
        padded = torch.from_numpy(padded)
        result = FoldedTensor(
            data=padded,
            lengths=lengths,
            data_dims=data_dims,
            full_names=full_names,
            indexer=indexer,
        )
    elif isinstance(data, torch.Tensor) and lengths is not None:
        np_indexer, shape = _C.make_refolding_indexer(lengths, data_dims)
        assert shape == list(data.shape[: len(data_dims)])
        result = FoldedTensor(
            data=data,
            lengths=lengths,
            data_dims=data_dims,
            full_names=full_names,
            indexer=torch.from_numpy(np_indexer).to(data.device),
        )
    else:
        raise ValueError(
            "as_folded_tensor expects:"
            "- a `data` (optionally nested) sequence"
            "- a `data_dims` sequence of names for padded dimensions"
            "- a `full_names` sequence of names for variable dimensions"
        )
    if device is not None:
        result = result.to(device)
    return result


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
        indexer: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        data_dims = data_dims
        full_names = full_names
        instance = data.as_subclass(cls)
        instance.lengths = lengths
        instance.data_dims = data_dims
        instance.full_names = full_names
        instance.indexer = indexer
        instance._mask = mask
        return instance

    def with_data(self, data: torch.Tensor):
        return FoldedTensor(
            data=data,
            lengths=self.lengths,
            data_dims=self.data_dims,
            full_names=self.full_names,
            indexer=self.indexer,
            mask=self._mask,
        )

    @property
    def mask(self):
        if self._mask is None:
            self._mask = torch.zeros(
                self.shape[: len(self.data_dims)],
                dtype=torch.bool,
                device=self.device,
            )
            self._mask.view(-1)[self.indexer] = True
        return self._mask

    def as_tensor(self):
        return self.as_subclass(torch.Tensor)

    def to(self, *args, **kwargs):
        with torch._C.DisableTorchFunction():
            result = super().to(*args, **kwargs)
            return FoldedTensor(
                data=result,
                lengths=self.lengths,
                data_dims=self.data_dims,
                full_names=self.full_names,
                indexer=self.indexer.to(result.device, copy=kwargs.get("copy", False)),
                mask=self._mask.to(result.device, copy=kwargs.get("copy", False))
                if self._mask is not None
                else None,
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super().__torch_function__(func, types, args, kwargs)

        if not isinstance(result, torch.Tensor):
            return result

        ft = None
        for arg in (*args, *(kwargs or {}).values()):
            if isinstance(arg, FoldedTensor):
                assert (
                    ft is None or ft.data_dims == arg.data_dims
                ), "Cannot perform operation on FoldedTensors with different structure"
                ft = arg
            if isinstance(arg, list):
                for item in arg:
                    assert (
                        ft is None or ft.data_dims == item.data_dims
                    ), "Cannot perform operation on FoldedTensors with different structure"
                    ft = item

        if ft.shape[: len(ft.data_dims)] != result.shape[: len(ft.data_dims)]:
            return result.as_subclass(torch.Tensor)

        result = FoldedTensor(
            data=result,
            lengths=ft.lengths,
            data_dims=ft.data_dims,
            full_names=ft.full_names,
            indexer=ft.indexer,
            mask=ft._mask,
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
