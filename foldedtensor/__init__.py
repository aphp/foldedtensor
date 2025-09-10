import typing
import warnings
from collections import UserList
from multiprocessing.reduction import ForkingPickler
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.autograd import Function

from . import _C  # type: ignore[import]

Dim = Union[int, str]


def map_indices(
    indices: Tuple[Sequence[int], ...],
    indice_dims: Tuple[int, ...],
    lengths: Sequence[Sequence[int]],
    data_dims: Tuple[int, ...],
    *,
    return_tensors: Optional[str] = None,
):
    """
    Compute leaf (last-dim) flat indices given indices in other dimensions.

    Parameters
    ----------
    indices: Tuple[Sequence[int], ...]
        Tuple of index sequences (broadcasted together) describing positions
        along `indice_dims`.
    indice_dims: Tuple[int, ...]
        Names or indices of the addressing dims.
    lengths: Sequence[Sequence[int]]
        Nested lengths describing the folded structure.
    data_dims: Tuple[int, ...]
        Names or indices describing the padded layout used for flattening.
    return_tensors: Optional[str], optional (default=None)
        Return type: "pt" for torch, "np" for numpy, "list" for python list.

    Returns
    -------
    Union[List[int], np.ndarray, torch.Tensor]
        Returns a list of flat indices compatible with `.view(-1)` of a tensor
        refolded with `data_dims`.
    """
    D = len(lengths)
    if data_dims[-1] != D - 1:
        raise ValueError(
            "data_dims must end with the last variable dimension (e.g., 'token')"
        )

    orig_shape = None
    saw_pt = False
    saw_np = False
    np_indices: Tuple[np.ndarray, ...] = tuple(
        (
            (
                lambda a: (
                    (lambda arr: arr.reshape(-1))(
                        a.detach().cpu().numpy()
                        if isinstance(a, torch.Tensor)
                        else (np.asarray(a))
                    )
                )
            )(arr)
        )
        for arr in indices
    )  # type: ignore[arg-type]

    # Track types and original shape from the first array
    first = indices[0]
    if isinstance(first, torch.Tensor):
        saw_pt = True
        orig_shape = tuple(first.shape)
    else:
        arr0 = np.asarray(first)
        if arr0.ndim > 1:
            orig_shape = tuple(arr0.shape)
        saw_np = isinstance(first, np.ndarray) or saw_np

    if len(indice_dims) != len(np_indices):
        raise ValueError("indices and indice_dims must have the same length")

    res = _C.map_indices(
        lengths,
        list(data_dims),
        list(indice_dims),
        np_indices,
    )
    out_np = np.asarray(res)
    # Reshape if needed
    if orig_shape is not None:
        out_np = out_np.reshape(orig_shape)

    if return_tensors == "pt" or return_tensors is None and saw_pt:
        return torch.from_numpy(out_np)
    if return_tensors == "np" or return_tensors is None and saw_np:
        return out_np
    return out_np.tolist()


def make_indices_ranges(
    *,
    begins,
    ends,
    indice_dims,
    lengths,
    data_dims,
    return_tensors: Union[typing.Optional[str], bool] = None,
):
    """
    Expand multiple ranges specified along indice_dims into:
    - flat indices (compatible with `.view(-1)` of a tensor refolded with `data_dims`),
    - start offsets per span,
    - and span indices (the span id for each expanded position).

    Parameters use the same conventions as map_indices. `begins` and `ends` are
    tuples of 1D tensors or lists corresponding to each dimension in `indice_dims`.
    Ranges are half-open: [begin, end), with boundary support when the last
    coordinate equals the number of children of its parent.
    """
    if not isinstance(begins, (list, tuple)) or not isinstance(ends, (list, tuple)):
        raise TypeError("begins and ends must be tuples/lists of arrays")
    if len(begins) != len(indice_dims) or len(ends) != len(indice_dims):
        raise ValueError("begins/ends must match indice_dims length")

    saw_pt = False
    saw_np = False
    # Determine original shape from the first begins entry
    first_b = begins[0]
    if isinstance(first_b, torch.Tensor):
        orig_shape = tuple(first_b.shape)
        saw_pt = True
    else:
        arr0 = np.asarray(first_b)
        orig_shape = tuple(arr0.shape) if arr0.ndim > 1 else None
        saw_np = isinstance(first_b, np.ndarray) or saw_np

    def _to_np1d(x):
        nonlocal saw_pt, saw_np
        if isinstance(x, torch.Tensor):
            saw_pt = True
            return x.detach().cpu().numpy().reshape(-1)
        a = np.asarray(x)
        if isinstance(x, np.ndarray):
            saw_np = True
        return a.reshape(-1)

    begins_np = [_to_np1d(b) for b in begins]
    ends_np = [_to_np1d(e) for e in ends]

    res = _C.make_indices_ranges(
        lengths,
        list(data_dims),
        list(indice_dims),
        begins_np,
        ends_np,
    )

    indices, offsets, span_indices = res
    indices_np = np.asarray(indices)
    offsets_np = np.asarray(offsets)
    span_indices_np = np.asarray(span_indices)

    # Reshape offsets to original input shape if multi-dimensional
    if orig_shape is not None:
        offsets_np = offsets_np.reshape(orig_shape)

    if return_tensors == "pt" or return_tensors is None and saw_pt:
        return (
            torch.from_numpy(indices_np.astype(np.int64, copy=False)),
            torch.from_numpy(offsets_np.astype(np.int64, copy=False)),
            torch.from_numpy(span_indices_np.astype(np.int64, copy=False)),
        )
    if return_tensors == "np" or return_tensors is None and saw_np:
        return (
            indices_np,
            offsets_np,
            span_indices_np,
        )
    return (
        indices_np.astype(np.int64, copy=False).tolist(),
        offsets_np.astype(np.int64, copy=False).tolist(),
        span_indices_np.astype(np.int64, copy=False).tolist(),
    )


np_to_torch_dtype = {
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

pass_through_functions = {
    torch.Tensor._grad.__get__,
    torch.Tensor.grad,
    torch.Tensor._base.__get__,
    torch.Tensor.__repr__,
    torch.Tensor.__str__,
    torch.Tensor.__format__,
    torch.Tensor.shape.__get__,
    torch.Tensor.size.__get__,
    torch.Tensor.dtype.__get__,
    torch.Tensor.device.__get__,
}
if hasattr(torch._C, "TensorBase"):
    pass_through_functions.add(torch._C.TensorBase.size)
else:
    pass_through_functions.add(torch.Tensor.size)

try:
    DisableTorchFunctionSubclass = torch._C.DisableTorchFunctionSubclass
except AttributeError:
    DisableTorchFunctionSubclass = torch._C.DisableTorchFunction

__version__ = "0.4.0"


class FoldedTensorLayout(UserList):
    """
    Folded tensor layout information.
    """

    def __init__(
        self,
        initlist: Optional[Sequence[Sequence[int]]] = None,
        *,
        data_dims: Optional[Sequence[Union[int, str]]],
        full_names: Optional[Sequence[str]],
    ) -> None:
        super().__init__(initlist or [])
        self._full_names: Optional[Tuple[str, ...]] = (
            tuple(full_names) if full_names is not None else None
        )
        if self._full_names is not None:
            dd = tuple(
                d if isinstance(d, int) else self._full_names.index(d)
                for d in data_dims
            )
        else:
            # Accept ints only when no names are provided
            dd = tuple(int(d) for d in data_dims)
        self._data_dims: Optional[Tuple[int, ...]] = dd

    def __hash__(self):
        return id(self)

    @property
    def full_names(self) -> Optional[Tuple[str, ...]]:
        return self._full_names

    @property
    def data_dims(self) -> Optional[Tuple[int, ...]]:
        return self._data_dims

    def __getitem__(self, index: Union[int, str]) -> typing.Any:
        if isinstance(index, str):
            if self._full_names is None:
                raise ValueError(
                    "Cannot resolve named index without full_names in the layout"
                )
            try:
                index = self._full_names.index(index)
            except ValueError as exc:  # pragma: no cover
                raise ValueError(f"Unknown dimension name {index!r}") from exc
        if not isinstance(index, int):  # pragma: no cover
            raise TypeError("Index must be an int or a str")
        return super().__getitem__(index)

    def resolve_dim(self, dim):
        if isinstance(dim, tuple):
            return tuple(self.resolve_dim(d) for d in dim)
        if isinstance(dim, str):
            if self._full_names is None:
                raise ValueError(
                    "Cannot resolve named dim without full_names in the layout"
                )
            try:
                dim = self._full_names.index(dim)
            except ValueError as exc:  # pragma: no cover
                raise ValueError(f"Unknown dimension name {dim!r}") from exc
        return int(dim)

    def map_indices(
        self,
        indices: Tuple[Sequence[int], ...],
        indice_dims: Tuple[Union[int, str], ...],
        *,
        data_dims: Optional[Sequence[Union[int, str]]] = None,
        return_tensors: Optional[str] = None,
    ):
        indice_dims = self.resolve_dim(indice_dims)
        data_dims = self.resolve_dim(data_dims or self.data_dims)

        return map_indices(
            indices=indices,
            indice_dims=indice_dims,
            lengths=self,
            data_dims=data_dims,
            return_tensors=return_tensors,
        )

    def make_indices_ranges(
        self,
        *,
        begins,
        ends,
        indice_dims,
        data_dims: Optional[Sequence[Union[int, str]]] = None,
        return_tensors: Optional[str] = None,
    ):
        # Resolve indice_dims against this layout's names if provided
        indice_dims = self.resolve_dim(indice_dims)
        data_dims = self.resolve_dim(data_dims or self.data_dims)

        return make_indices_ranges(
            begins=begins,
            ends=ends,
            indice_dims=indice_dims,
            lengths=self,
            data_dims=data_dims,
            return_tensors=return_tensors,
        )


# Backward-compatibility alias
FoldedTensorLengths = FoldedTensorLayout


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
        ctx.input_indexer = self.indexer

        device = self.data.device

        np_new_indexer, shape_prefix = _C.make_refolding_indexer(
            self.lengths,
            dims,
        )
        data = self.as_tensor()
        indexer = torch.from_numpy(np_new_indexer).to(device)
        ctx.output_indexer = indexer
        shape_suffix = data.shape[len(self.data_dims) :]
        refolded_data = torch.zeros(
            (*shape_prefix, *shape_suffix), dtype=data.dtype, device=device
        )
        refolded_data.view(-1, *shape_suffix)[indexer] = data.view(
            -1, *shape_suffix
        ).index_select(0, self.indexer)
        lengths = FoldedTensorLayout(
            self.lengths, data_dims=dims, full_names=self.full_names
        )
        return FoldedTensor(
            data=refolded_data,
            lengths=lengths,
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
        shape_suffix = grad_output.shape[len(ctx.new_data_dims) :]
        grad_input = torch.zeros(
            (*shape_prefix, *shape_suffix), dtype=grad_output.dtype, device=device
        )
        grad_input.view(-1, *shape_suffix)[ctx.input_indexer] = grad_output.reshape(
            -1, *shape_suffix
        ).index_select(0, ctx.output_indexer)
        return grad_input, None


type_to_dtype_dict = {
    int: torch.tensor([0]).dtype,
    float: torch.tensor([0.0]).dtype,
    bool: torch.bool,
    None: torch.tensor([0.0]).dtype,
}


def get_metadata(nested_data):
    item = None
    deepness = 0

    def rec(seq, depth=0):
        nonlocal item, deepness
        if isinstance(seq, (list, tuple)):
            depth += 1
            deepness = max(deepness, depth)
            for item in seq:
                yield from rec(item, depth)
        else:
            yield

    next(rec(nested_data), 0)
    return deepness, type(item)


def as_folded_tensor(
    data: Sequence,
    data_dims: Optional[Sequence[Union[int, str]]] = None,
    full_names: Optional[Sequence[str]] = None,
    dtype: Optional[torch.dtype] = None,
    lengths: Optional[Union[FoldedTensorLayout, List[List[int]]]] = None,
    device: Optional[Union[str, torch.device]] = None,
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
    device: Optional[Unit[str, torch.device]]
        The device of the output tensor
    """
    if isinstance(lengths, FoldedTensorLayout):
        data_dims = lengths.data_dims or data_dims
        full_names = lengths.full_names or full_names
    if full_names is not None:
        if data_dims is not None:
            data_dims = tuple(
                dim if isinstance(dim, int) else full_names.index(dim)
                for dim in data_dims
            )
            if (data_dims[-1] + 1) != len(full_names):
                raise ValueError(
                    "The last dimension of `data_dims` must be the last "
                    "variable dimension."
                )
        elif full_names is not None:
            data_dims = tuple(range(len(full_names)))
    if isinstance(data, torch.Tensor) and lengths is not None:
        data_dims = data_dims or tuple(range(len(lengths)))
        np_indexer, shape = _C.make_refolding_indexer(lengths, data_dims)
        assert shape == list(data.shape[: len(data_dims)]), (
            f"Shape inferred from lengths is not compatible with data dims: {shape}, "
            f"{data.shape}, {len(data_dims)}"
        )
        layout = FoldedTensorLayout(lengths, data_dims=data_dims, full_names=full_names)
        result = FoldedTensor(
            data=data,
            lengths=layout,
            indexer=torch.from_numpy(np_indexer).to(data.device),
        )
    elif isinstance(data, Sequence):
        # if dtype is None:
        #     raise ValueError("dtype must be provided when `data` is a sequence")
        if data_dims is None or dtype is None:
            deepness, inferred_dtype = get_metadata(data)
        else:
            deepness = len(full_names) if full_names is not None else len(data_dims)
        if data_dims is None:
            data_dims = tuple(range(deepness))
        if dtype is None:
            dtype = type_to_dtype_dict.get(inferred_dtype)
        dtype = np_to_torch_dtype.get(dtype, dtype)
        padded, indexer, lengths = _C.nested_py_list_to_padded_array(
            data,
            data_dims,
            np.dtype(dtype),
        )
        indexer = torch.from_numpy(indexer)
        padded = torch.from_numpy(padded)
        # In case of empty sequences, lengths are not computed correctly
        lengths = (list(lengths) + [[0]] * deepness)[:deepness]
        layout = FoldedTensorLayout(lengths, data_dims=data_dims, full_names=full_names)
        result = FoldedTensor(
            data=padded,
            lengths=layout,
            indexer=indexer,
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


def _postprocess_func_result(result, input):
    if (
        input is not None
        and input.shape[: len(input.data_dims)] != result.shape[: len(input.data_dims)]
    ):
        return result

    return FoldedTensor(
        data=result,
        lengths=input.lengths,
        indexer=input.indexer,
        mask=input._mask,
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
        lengths: FoldedTensorLayout,
        indexer: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        instance = data.as_subclass(cls)
        instance.lengths = lengths
        instance.indexer = indexer
        instance._mask = mask
        return instance

    def with_data(self, data: torch.Tensor):
        return FoldedTensor(
            data=data,
            lengths=self.lengths,
            indexer=self.indexer,
            mask=self._mask,
        )

    @property
    def data_dims(self) -> Tuple[int, ...]:
        return self.lengths.data_dims

    @property
    def full_names(self) -> Optional[Tuple[str, ...]]:
        return self.lengths.full_names

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
            res = super().to(*args, **kwargs)
            copy = kwargs.get("copy", False)
            nb = kwargs.get("non_blocking", False)
            return FoldedTensor(
                data=res,
                lengths=self.lengths,
                indexer=self.indexer.to(res.device, copy=copy, non_blocking=nb),
                mask=self._mask.to(res.device, copy=copy, non_blocking=nb)
                if self._mask is not None
                else None,
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in pass_through_functions:
            with DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        with DisableTorchFunctionSubclass():
            result = func(*args, **kwargs)

        if func is torch.Tensor.share_memory_:
            self = args[0]
            self.indexer.share_memory_()
            if self._mask is not None:
                self._mask.share_memory_()
            return self

        ft = None
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, FoldedTensor):
                assert (
                    ft is None or ft.data_dims == arg.data_dims
                ), "Cannot perform operation on FoldedTensors with different layouts"
                ft = arg
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, FoldedTensor):
                        assert ft is None or ft.data_dims == item.data_dims, (
                            "Cannot perform operation on FoldedTensors with "
                            "different layouts"
                        )
                        ft = item

        if isinstance(result, torch.Tensor):
            return _postprocess_func_result(result, ft)

        if (
            isinstance(result, (tuple, list))
            and len(result)
            and isinstance(result[0], torch.Tensor)
        ):
            return type(result)(
                _postprocess_func_result(item, ft)
                if isinstance(item, FoldedTensor)
                else item
                for item in result
            )

        return result

    def clone(self):
        cloned = super().clone()
        cloned.indexer = self.indexer.clone()
        if self._mask is not None:
            cloned._mask = self._mask.clone()
        return cloned

    def refold(self, *dims: Union[Sequence[Union[int, str]], int, str]):
        if not isinstance(dims[0], (int, str)):
            assert len(dims) == 1, (
                "Expected the first only argument to be a "
                "sequence or each arguments to be ints or strings"
            )
            dims = dims[0]
        try:
            dims = tuple(
                dim if isinstance(dim, int) else self.full_names.index(dim)
                for dim in dims
            )
        except ValueError:  # pragma: no cover
            raise ValueError(
                f"Folded tensor with available dimensions {self.full_names} "
                f"could not be refolded with dimensions {list(dims)}"
            )

        # Ensure the leaf (last variable) dimension is last in the refolded layout
        leaf = len(self.lengths) - 1
        if dims[-1] != leaf:
            leaf_name = (
                self.full_names[leaf] if self.full_names is not None else str(leaf)
            )
            dim_names = tuple(
                self.full_names[d] if self.full_names is not None else str(d)
                for d in dims
            )
            raise ValueError(
                "The last dimension of data_dims must be the last variable "
                f"dimension {leaf_name!r} (ie. {leaf}); got data_dims={dim_names} "
                f"(ie. {tuple(dims)}"
            )

        if dims == self.data_dims:
            return self

        return Refold.apply(self, dims)


def reduce_foldedtensor(self: FoldedTensor):
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=UserWarning,
        )
        return (
            FoldedTensor,
            (
                self.data.as_tensor(),
                self.lengths,
                self.indexer.clone()
                if self.indexer.is_shared() and self.indexer.storage().is_cuda
                else self.indexer,
                None
                if self._mask is not None
                and self._mask.is_shared()
                and self._mask.storage().is_cuda
                else self._mask,
            ),
        )


ForkingPickler.register(FoldedTensor, reduce_foldedtensor)
