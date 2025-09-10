<div align="center">
<p align="center">
  <img src="https://github.com/aphp/foldedtensor/raw/main/banner.png" width="70%">
</p>
</div>
<br/>

---

![Tests](https://img.shields.io/github/actions/workflow/status/aphp/foldedtensor/tests.yml?branch=main&label=tests&style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/foldedtensor?color=blue&style=flat-square)](https://pypi.org/project/foldedtensor/)
[![Coverage](https://raw.githubusercontent.com/aphp/foldedtensor/coverage/coverage.svg)](https://raw.githubusercontent.com/aphp/foldedtensor/coverage/coverage.txt)
[![License](https://img.shields.io/github/license/aphp/foldedtensor?color=x&style=flat-square)](https://github.com/aphp/foldedtensor/blob/main/LICENSE)
![PyPI - Downloads](https://img.shields.io/pypi/dm/foldedtensor?style=flat-square&color=purple)
![Python versions](https://img.shields.io/pypi/pyversions/foldedtensor?style=flat-square)

# FoldedTensor: PyTorch extension for handling deeply nested sequences of variable length

`foldedtensor` is a PyTorch extension that provides efficient handling of tensors containing deeply nested sequences variable sizes. It enables the flattening/unflattening (or unfolding/folding) of data dimensions based on a inner structure of sequence lengths. This library is particularly useful when working with data that can be split in different ways and enables you to avoid choosing a fixed representation.

## Installation

The library can be installed with pip:

```bash
pip install foldedtensor
```

## Features

- Support for arbitrary numbers of nested dimensions
- No computational overhead when dealing with already padded tensors
- Dynamic re-padding (or refolding) of data based on stored inner lengths
- Automatic mask generation and updating whenever the tensor is refolded
- C++ optimized code for fast data loading from Python lists and refolding
- Flexibility in data representation, making it easy to switch between different layouts when needed

## Examples

At its simplest, `foldedtensor` can be used to convert nested Python lists into a PyTorch tensor:

```python
from foldedtensor import as_folded_tensor

ft = as_folded_tensor(
    [
        [0, 1, 2],
        [3],
    ],
)
# FoldedTensor([[0, 1, 2],
#               [3, 0, 0]])
```

You can also specify names and flattened/unflattened dimensions at the time of creation:

```python
import torch
from foldedtensor import as_folded_tensor

# Creating a folded tensor from a nested list
# There are 2 samples, the first with 5 lines, the second with 1 line.
# Each line contain between 1 and 2 words.
ft = as_folded_tensor(
    [
        [[1], [], [], [], [2, 3]],
        [[4, 3]],
    ],
    data_dims=("samples", "words"),
    full_names=("samples", "lines", "words"),
    dtype=torch.long,
)
print(ft)
# FoldedTensor([[1, 2, 3],
#               [4, 3, 0]])
```

Once created, you can change the shape of the tensor by refolding it:

```python
# Refold on the lines and words dims (flatten the samples dim)
print(ft.refold(("lines", "words")))
# FoldedTensor([[1, 0],
#               [0, 0],
#               [0, 0],
#               [0, 0],
#               [2, 3],
#               [4, 3]])

# Refold on the words dim only: flatten everything
print(ft.refold(("words",)))
# FoldedTensor([1, 2, 3, 4, 3])
```

The tensor can be further used with standard PyTorch operations:

```python
# Working with PyTorch operations
embedder = torch.nn.Embedding(10, 16)
embedding = embedder(ft.refold(("words",)))
print(embedding.shape)
# torch.Size([5, 16]) # 5 words total, 16 dims

refolded_embedding = embedding.refold(("samples", "words"))
print(refolded_embedding.shape)
# torch.Size([2, 5, 16]) # 2 samples, 5 words max, 16 dims
```

### Pooling spans

You can pool variable length spans directly on a refolded view without padding by building flat indices and offsets and then using `embedding_bag`.

The helper `lengths.make_indices_ranges` expands ranges defined over one or more variable dimensions.

- `indices` are the flat positions in the refolded tensor viewed as a single dimension
- `offsets` are the start positions of each span within `indices`
- `spans` gives the span id for every expanded position, which can be useful for functions like `torch.index_add` or `torch.index_reduce`

Example that sums over word spans to produce one vector per span

```python
import torch
import foldedtensor as ft

# Build a 4 level tensor with names: first word of the first context is split into three tokens, etc
input_ids = ft.as_folded_tensor(
    [
        [
            [[0, 2, 3], [10], [4]],
            [[0, 1, 2], [2, 3], [10, 11], [100, 101]],
        ],
    ],
    full_names=("sample", "context", "word", "token"),
).refold(
    "token"
)  # any refolding is fine

# Create embeddings from the input ids
embedding = torch.nn.Embedding(2048, 16)
weight = embedding(input_ids)

# Pool two word spans per the test
# span 1 covers words 0 to 2 -> mean pool over 4 tokens [0, 2, 3, 10]
# span 2 covers words 5 to 7 -> mean pool over 4 tokens [10, 11, 100, 101]
indices, offsets, spans = input_ids.lengths.make_indices_ranges(
    begins=(torch.tensor([0, 5]),),
    ends=(torch.tensor([2, 7]),),
    indice_dims=("word",),
)

# Sum embeddings over each span
pooled = torch.nn.functional.embedding_bag(
    input=indices,
    # Flatten embeddings so rows align with flattened token positions
    weight=weight.view(-1, weight.size(-1)),
    offsets=offsets,
    mode="mean",
)
print(pooled)
```

## Benchmarks

View the comparisons of `foldedtensor` against various alternatives here: [docs/benchmarks](https://github.com/aphp/foldedtensor/blob/main/docs/benchmark.md).

## Comparison with alternatives

Unlike other ragged or nested tensor implementations, a FoldedTensor does not enforce a specific structure on the nested data, and does not require padding all dimensions. This provides the user with greater flexibility when working with data that can be arranged in multiple ways depending on the data transformation. Moreover, the C++ optimization ensures high performance, making it ideal for handling deeply nested tensors efficiently.

Here is a comparison with other common implementations for handling nested sequences of variable length:

| Feature                   | NestedTensor | MaskedTensor | FoldedTensor |
|---------------------------|--------------|--------------|--------------|
| Inner data structure      | Flat         | Padded       | Arbitrary    |
| Max nesting level         | 1            | 1            | ∞            |
| From nested python lists  | No           | No           | Yes          |
| Layout conversion         | To padded    | No           | Any          |
| Reduction ops w/o padding | Yes          | No           | No           |
