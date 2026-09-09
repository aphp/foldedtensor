import warnings
from collections import UserList
from multiprocessing.reduction import ForkingPickler
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.autograd import Function

from . import _C  # type: ignore[import]


def make_indices_ranges(
    *, begins, ends, indice_dims, lengths, data_dims, return_tensors=None
):
    """
    Expand half open ranges into storage indices for span pooling, excluding padding

    Parameters
    ----------
    begins, ends
        Coordinate arrays broadcast together, one per addressed dimension
    indice_dims
        Increasing dimension indices used to address range boundaries
    lengths
        Lengths of nested sequences at each dimension
    data_dims
        Dimensions represented as tensor axes, ending with the innermost dimension
    return_tensors
        Output format, pt, np or list, inferred from the input when omitted

    Returns
    -------
    indices, offsets, span_indices
        Flat storage indices excluding padding, start offsets with the broadcast
        input shape and the span index of each selected element

    Examples
    --------
    Expand two overlapping ranges and an empty range in a sequence of six elements

    ```python
    indices, offsets, span_indices = make_indices_ranges(
        begins=([0, 2, 5],),
        ends=([3, 4, 5],),
        indice_dims=(0,),
        lengths=[[6]],
        data_dims=(0,),
    )
    # indices:      [0, 1, 2, 2, 3]
    #                -------  ----
    # offsets:      [0,       3,    5]
    # span_indices: [0, 0, 0, 1, 1]
    ```
    """
    if (
        not indice_dims
        or len(begins) != len(indice_dims)
        or len(ends) != len(indice_dims)
    ):
        raise ValueError("begins and ends must match the nonempty indice_dims")
    coords = np.broadcast_arrays(
        *[
            np.asarray(
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x,
                dtype=np.int64,
            )
            for x in (*begins, *ends)
        ]
    )
    shape = coords[0].shape
    coords = np.stack(coords).reshape(2 * len(indice_dims), -1)
    indices, offsets, spans = _C.make_indices_ranges(
        coords[: len(indice_dims)],
        coords[len(indice_dims) :],
        indice_dims,
        lengths,
        data_dims,
    )
    result = (indices, offsets.reshape(shape), spans)
    if (
        return_tensors == "pt"
        or return_tensors is None
        and isinstance(begins[0], torch.Tensor)
    ):
        return tuple(
            torch.as_tensor(
                x,
                device=begins[0].device
                if isinstance(begins[0], torch.Tensor)
                else None,
            )
            for x in result
        )
    if (
        return_tensors == "np"
        or return_tensors is None
        and isinstance(begins[0], np.ndarray)
    ):
        return result
    return tuple(x.tolist() for x in result)


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

__version__ = "0.5.0.dev1"


class FoldedTensorLayout(UserList):
    """
    Sequence lengths and tensor dimensions used by refolding and range expansion

    Parameters
    ----------
    initlist: Sequence[Sequence[int]]
        Lengths of nested sequences at each dimension, ordered by their parents
    data_dims: Optional[Sequence[Union[int, str]]]
        Dimensions represented as tensor axes, by index or name, defaults to all
        dimensions, omitted dimensions are flattened into the following axis
    full_names: Optional[Sequence[str]]
        Names of all nested dimensions, used to address lengths and ranges by name
    """

    def __init__(self, initlist=(), *, data_dims=None, full_names=None):
        super().__init__(initlist)
        self.full_names = tuple(full_names) if full_names is not None else None
        self.data_dims = (
            self.resolve_dims(data_dims)
            if data_dims is not None
            else tuple(range(len(self)))
        )

    def __setstate__(self, state):
        """
        Restore sequence lengths with defaults for omitted layout metadata
        """
        self.__dict__.update(
            full_names=None, data_dims=tuple(range(len(state["data"])))
        )
        self.__dict__.update(state)

    def __hash__(self):
        return id(self)

    def __getitem__(self, index):
        return self.data[
            self.full_names.index(index) if isinstance(index, str) else index
        ]

    def resolve_dims(self, dims):
        """
        Resolve dimension names for layout construction and index mapping
        """
        return tuple(
            self.full_names.index(d) if isinstance(d, str) else d for d in dims
        )

    def make_indices_ranges(
        self, *, begins, ends, indice_dims, data_dims=None, return_tensors=None
    ):
        """
        Expand ranges into storage indices and span offsets for pooling

        Parameters
        ----------
        begins, ends: Sequence[array-like]
            Start and end coordinates, one array per addressed dimension, broadcast
            together, end coordinates are excluded
        indice_dims: Sequence[Union[int, str]]
            Increasing dimension indices or names used to address range boundaries,
            coordinates in omitted dimensions are flattened
        data_dims: Optional[Sequence[Union[int, str]]]
            Dimensions represented as tensor axes, defaults to this layout,
            must end with the innermost dimension
        return_tensors: Optional[str]
            Output format, pt, np or list, inferred from the first begin coordinate
            array when omitted, tensor outputs use the device of the first begin
            coordinate if it is a tensor, or CPU otherwise

        Returns
        -------
        indices
            Flat storage indices excluding padding, repeated for overlapping ranges
        offsets
            Start offset of each range in indices, with the broadcast coordinate shape,
            empty ranges have repeated offsets
        span_indices
            Range index for each selected element, using flattened broadcast order
        """
        return make_indices_ranges(
            begins=begins,
            ends=ends,
            indice_dims=self.resolve_dims(indice_dims),
            lengths=self,
            data_dims=self.data_dims
            if data_dims is None
            else self.resolve_dims(data_dims),
            return_tensors=return_tensors,
        )


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
        data_dims = lengths.data_dims if data_dims is None else data_dims
        full_names = lengths.full_names if full_names is None else full_names
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
        Embedding values or scalar data
    lengths: Union[FoldedTensorLayout, List[List[int]]]
        Lengths of nested sequences at each dimension with optional layout metadata
    data_dims: Optional[Sequence[int]]
        Tensor axes ending with the innermost dimension, defaults to the
        layout dimensions or all dimensions for plain lengths
    full_names: Optional[Sequence[str]]
        Names of the nested dimensions, defaults to the layout names
    indexer: torch.Tensor
        One flat storage row index per element of the innermost dimension,
        excluding padding
    mask: Optional[torch.Tensor]
        Boolean mask identifying storage rows without padding, computed lazily
    """

    def __new__(
        cls,
        data: torch.Tensor,
        lengths: Union[FoldedTensorLayout, List[List[int]]],
        data_dims: Optional[Sequence[int]] = None,
        full_names: Optional[Sequence[str]] = None,
        indexer: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if indexer is None:
            raise TypeError("FoldedTensor requires an indexer")
        if (
            not isinstance(lengths, FoldedTensorLayout)
            or data_dims is not None
            or full_names is not None
        ):
            lengths = FoldedTensorLayout(
                lengths,
                data_dims=data_dims
                if data_dims is not None
                else getattr(lengths, "data_dims", None),
                full_names=full_names
                if full_names is not None
                else getattr(lengths, "full_names", None),
            )
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

    @data_dims.setter
    def data_dims(self, dims):
        self.lengths = FoldedTensorLayout(
            self.lengths, data_dims=dims, full_names=self.full_names
        )

    @property
    def full_names(self) -> Optional[Tuple[str, ...]]:
        return self.lengths.full_names

    @full_names.setter
    def full_names(self, names):
        self.lengths = FoldedTensorLayout(
            self.lengths, data_dims=self.data_dims, full_names=names
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
            copy = kwargs.get("copy", False)
            non_blocking = kwargs.get("non_blocking", False)
            return FoldedTensor(
                data=result,
                lengths=self.lengths,
                indexer=self.indexer.to(
                    result.device, copy=copy, non_blocking=non_blocking
                ),
                mask=self._mask.to(result.device, copy=copy, non_blocking=non_blocking)
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
                assert ft is None or ft.data_dims == arg.data_dims, (
                    "Cannot perform operation on FoldedTensors with "
                    "different structures"
                )
                ft = arg
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, FoldedTensor):
                        assert ft is None or ft.data_dims == item.data_dims, (
                            "Cannot perform operation on FoldedTensors with "
                            "different structures"
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
        except ValueError:
            raise ValueError(
                f"Folded tensor with available dimensions {self.full_names} "
                f"could not be refolded with dimensions {list(dims)}"
            )

        if not dims or dims[-1] != len(self.lengths) - 1:
            raise ValueError(
                "The last dimension of data_dims must be the last variable dimension"
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
                self.data_dims,
                self.full_names,
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
