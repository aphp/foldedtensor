
Benchmarks
----------

This file was generated from [`scripts/benchmark.py`](../scripts/benchmark.py).

It compares the performance of `foldedtensor` with various alternatives for padding
and working with nested lists and tensors.

Environment:
- `torch.__version__ == '2.8.0'`
- `foldedtensor.__version__ == '0.4.0'`
- `python == 3.11.3`
- `sys.platform == 'darwin'`


## Case 1 (pad variable lengths nested list)

The following 3-levelled nested lists has lengths of 32, then between 50 and 100, and then between 25 and 30.
```python
nested_list = make_nested_list(32, (50, 100), (25, 30), value=1)

Comparisons:
%timeit python_padding(nested_list)
# 100 loops, best of 5: 19.02 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 0.82 ms per loop

```
Speedup against best alternative: **23.24x** :rocket:

## Case 2 (same lengths nested lists)

```python
nested_list = make_nested_list(32, 100, 30, value=1)

%timeit torch.tensor(nested_list)
# 100 loops, best of 5: 7.86 ms per loop

%timeit torch.LongTensor(nested_list)
# 100 loops, best of 5: 3.69 ms per loop

%timeit python_padding(nested_list)
# 100 loops, best of 5: 23.35 ms per loop

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 3.94 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 1.18 ms per loop

```
Speedup against best alternative: **3.12x** :rocket:

## Case 3 (simple list)

```python
simple_list = make_nested_list(10000, value=1)

%timeit torch.tensor(simple_list)
# 100 loops, best of 5: 0.77 ms per loop

%timeit torch.LongTensor(simple_list)
# 100 loops, best of 5: 0.37 ms per loop

%timeit python_padding(simple_list)
# 100 loops, best of 5: 0.37 ms per loop

%timeit foldedtensor.as_folded_tensor(simple_list)
# 100 loops, best of 5: 0.10 ms per loop

```
Speedup against best alternative: **3.59x** :rocket:

## Case 4 (same lengths nested lists to flat tensor)

```python
nested_list = make_nested_list(32, 100, 30, value=1)

%timeit torch.tensor(nested_list).view(-1)
# 100 loops, best of 5: 7.83 ms per loop

%timeit torch.LongTensor(nested_list).view(-1)
# 100 loops, best of 5: 3.68 ms per loop

%timeit python_padding(nested_list).view(-1)
# 100 loops, best of 5: 23.17 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list).view(-1)
# 100 loops, best of 5: 1.19 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list, data_dims=(2,))
# 100 loops, best of 5: 1.16 ms per loop

```
Speedup against best alternative: **3.10x** :rocket:
## Case 5 (variable lengths nested lists) to padded embeddings

Nested lists with different lengths (second level lists have lengths between 50 and 150). We compare `foldedtensor` with `torch.nested`.
```python
nested_list = make_nested_list(32, (50, 150), 30, value=1)

# Padding with 0

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 4.40 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list).as_tensor()
# 100 loops, best of 5: 1.29 ms per loop

```
Speedup against best alternative: **3.41x** :rocket:
```python
# Padding with 1

%timeit torch.nested.nested_tensor([torch.FloatTensor(sub) for sub in nested_list]).to_padded_tensor(1)
# 100 loops, best of 5: 4.77 ms per loop

%timeit x = foldedtensor.as_folded_tensor(nested_list); x.masked_fill_(x.mask, 1)
# 100 loops, best of 5: 1.65 ms per loop

```
Speedup against best alternative: **2.89x** :rocket:

## Case 6 (2d padding)

```python
nested_list = make_nested_list(160, (50, 150), value=1)

%timeit python_padding(nested_list)
# 100 loops, best of 5: 1.73 ms per loop

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 1.48 ms per loop

%timeit torch.nn.utils.rnn.pad_sequence([torch.LongTensor(sub) for sub in nested_list], batch_first=True, padding_value=0)
# 100 loops, best of 5: 1.22 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 0.18 ms per loop

```
Speedup against best alternative: **6.68x** :rocket:

## Case 7 (summing vectors inside each differently-sized sequence, all concatenated)

```python
def sum_all_words_per_sample(t):
    begins = torch.arange(len(t.lengths[1]))
    ends = begins + 1
    indices, offsets, spans = t.lengths.make_indices_ranges(
        begins=(begins,), ends=(ends,), indice_dims=(0,)
    )
    return torch.nn.functional.embedding_bag(
        input=indices,
        weight=t.view(-1, t.size(-1)),
        offsets=offsets,
        mode="sum",
    )

embedder = torch.nn.Embedding(500, 128)
nested_list = make_nested_list(320, (150, 250), value=1)
ft = foldedtensor.as_folded_tensor(nested_list).refold(1)
ft = embedder(ft)


%timeit ft.refold(0, 1).sum(-2)
# 100 loops, best of 5: 3.54 ms per loop

%timeit sum_all_words_per_sample(ft)
# 100 loops, best of 5: 1.01 ms per loop

```
Speedup against pad-then-sum: **3.52x** :rocket:
