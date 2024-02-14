# ruff: noqa: F401, E501
import contextlib
import random
import warnings
from timeit import Timer

import torch
import torch.nested
import torch.nn.utils.rnn

import foldedtensor  # noqa: F401

warnings.filterwarnings("ignore")

torch.set_default_device("cpu")


def pad_tensors(tensors):
    """
    Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

    The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
    where `Si` is the maximum value of dimension `i` amongst all tensors.
    """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, : size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, : size[0], : size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, : size[0], : size[1], : size[2]] = tensor
        else:
            raise ValueError("Padding is supported for upto 3D tensors at max.")
    return padded_tensor


def python_padding(ints):
    """
    Converts a nested list of integers to a padded tensor.
    """
    if isinstance(ints, torch.Tensor):
        return ints
    if isinstance(ints, list):
        if isinstance(ints[0], int):
            return torch.LongTensor(ints)
        if isinstance(ints[0], torch.Tensor):
            return pad_tensors(ints)
        if isinstance(ints[0], list):
            return python_padding([python_padding(inti) for inti in ints])


def make_nested_list(arg, *rest, value):
    size = random.randint(*arg) if isinstance(arg, tuple) else arg
    if not rest:
        return [value] * size
    return [make_nested_list(*rest, value=value) for _ in range(size)]


def exec_and_print(code):
    print(code)
    print()
    exec(code, globals(), globals())


@contextlib.contextmanager
def block_code():
    print("```python")
    yield
    print("```")


def timeit(stmt, number=100, repeat=5):
    t = Timer(stmt, globals=globals())

    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        callback = None

        try:
            number, _ = t.autorange(callback)
        except:
            t.print_exc()
            return 1

    try:
        raw_timings = t.repeat(repeat, number)
    except Exception:
        t.print_exc()
        return 1

    def format_time(dt):
        return f"{dt * 1000:.2f} ms"

    timings = [dt / number for dt in raw_timings]

    best = min(timings)
    print("%timeit " + stmt)
    print(
        "# %d loop%s, best of %d: %s per loop\n"
        % (number, "s" if number != 1 else "", repeat, format_time(best))
    )


print(
    f"""
Benchmarks
----------

This file was generated from [`scripts/benchmark.py`](../scripts/benchmark.py).

It compares the performance of `foldedtensor` with various alternatives for padding
and working with nested lists and tensors.

Versions:
- `torch.__version__ == {torch.__version__!r}`
- `foldedtensor.__version__ == {foldedtensor.__version__!r}`
"""
)

if __name__ == "__main__":
    # fmt: off
    cases = [1, 2, 3, 4, 5, 6]
    if 1 in cases:
        print("\n## Case 1 (pad variable lengths nested list)\n")

        print("The following 3-levelled nested lists has lengths of 32, then "
              "between 50 and 100, and then between 25 and 30.")

        with block_code():
            exec_and_print("nested_list = make_nested_list(32, (50, 100), (25, 30), value=1)")

            print("Comparisons:")
            timeit("python_padding(nested_list)")
            timeit("foldedtensor.as_folded_tensor(nested_list)")

    if 2 in cases:
        print("\n## Case 2 (same lengths nested lists)\n")

        with block_code():
            exec_and_print("nested_list = make_nested_list(32, 100, 30, value=1)")
            timeit("torch.tensor(nested_list)")
            timeit("torch.LongTensor(nested_list)")
            timeit("python_padding(nested_list)")
            timeit("torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)")
            timeit("foldedtensor.as_folded_tensor(nested_list)")

    if 3 in cases:
        print("\n## Case 3 (simple list)\n")

        with block_code():
            exec_and_print("simple_list = make_nested_list(10000, value=1)")
            timeit("torch.tensor(simple_list)")
            timeit("torch.LongTensor(simple_list)")
            timeit("python_padding(simple_list)")
            timeit("foldedtensor.as_folded_tensor(simple_list)")

    if 4 in cases:
        print("\n## Case 4 (same lengths nested lists to flat tensor)\n")

        with block_code():
            exec_and_print("nested_list = make_nested_list(32, 100, 30, value=1)")
            timeit("torch.tensor(nested_list).view(-1)")
            timeit("torch.LongTensor(nested_list).view(-1)")
            timeit("python_padding(nested_list).view(-1)")
            timeit("foldedtensor.as_folded_tensor(nested_list).view(-1)")
            timeit("foldedtensor.as_folded_tensor(nested_list, data_dims=(2,))")

    if 5 in cases:
        print("## Case 5 (variable lengths nested lists) to padded embeddings\n")
        print("Nested lists with different lengths (second level lists have lengths "
              "between 50 and 150). We compare `foldedtensor` with `torch.nested`.")

        with block_code():
            exec_and_print("nested_list = make_nested_list(32, (50, 150), 30, value=1)")

            print("# Padding with 0\n")

            timeit("torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)")
            timeit("foldedtensor.as_folded_tensor(nested_list).as_tensor()")

            print("# Padding with 1\n")
            timeit("torch.nested.nested_tensor([torch.FloatTensor(sub) for sub in nested_list]).to_padded_tensor(1)")
            timeit("x = foldedtensor.as_folded_tensor(nested_list); x.masked_fill_(x.mask, 1)")

    if 6 in cases:
        print("\n## Case 6 (2d padding)\n")

        with block_code():
            exec_and_print("nested_list = make_nested_list(160, (50, 150), value=1)")

            timeit("python_padding(nested_list)")
            timeit("torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)")
            timeit(
                "torch.nn.utils.rnn.pad_sequence([torch.LongTensor(sub) for sub in nested_list], batch_first=True, padding_value=0)")
            timeit("foldedtensor.as_folded_tensor(nested_list)")

    if 7 in cases:

        def sum_all_words_per_sample(ft):
            lengths = ft.lengths
            ids = torch.arange(lengths[0][0])
            for i in range(1, len(lengths)):
                ids = torch.repeat_interleave(
                    ids,
                    lengths[i],
                    output_size=len(lengths[i + 1])
                    if i < len(lengths) - 1
                    else ft.size(len(ft.data_dims) - 1),
                )

            out = torch.zeros(lengths[0][0], ft.shape[-1])
            out.index_add_(source=ft.as_tensor(), dim=0, index=ids)

            return out


        print("\n## Case 7 (flat sums)\n")

        with block_code():
            exec_and_print(
                "embedder = torch.nn.Embedding(500, 128)\n"
                "nested_list = make_nested_list(320, (150, 250), value=1)\n"
                "ft = foldedtensor.as_folded_tensor(nested_list).refold(2)\n"
                "nt = torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list])\n"
                "ft = embedder(ft)\n"
                "nt = embedder(nt)\n"
            )

            timeit("nt.sum(dim=1)")
            timeit("sum_all_words_per_sample(ft)")

        # timeit("embedder(ft)")
        # timeit("embedder(ft).refold(0, 1)")
        # timeit("embedder(nt)")
    # fmt: on
