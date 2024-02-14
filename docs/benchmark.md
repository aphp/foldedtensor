
Benchmarks
----------

This file was generated from [`scripts/benchmark.py`](../scripts/benchmark.py).

It compares the performance of `foldedtensor` with various alternatives for padding
and working with nested lists and tensors.

Versions:
- `torch.__version__ == '2.0.1'`
- `foldedtensor.__version__ == '0.3.3'`


## Case 1 (pad variable lengths nested list)

The following 3-levelled nested lists has lengths of 32, then between 50 and 100, and then between 25 and 30.
```python
nested_list = make_nested_list(32, (50, 100), (25, 30), value=1)

Comparisons:
%timeit python_padding(nested_list)
# 100 loops, best of 5: 13.24 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 0.63 ms per loop

```


## Case 2 (same lengths nested lists)

```python
nested_list = make_nested_list(32, 100, 30, value=1)

%timeit torch.tensor(nested_list)
# 100 loops, best of 5: 6.44 ms per loop

%timeit torch.LongTensor(nested_list)
# 100 loops, best of 5: 2.64 ms per loop

%timeit python_padding(nested_list)
# 100 loops, best of 5: 16.68 ms per loop

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 2.90 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 0.96 ms per loop

```


## Case 3 (simple list)

```python
simple_list = make_nested_list(10000, value=1)

%timeit torch.tensor(simple_list)
# 100 loops, best of 5: 0.65 ms per loop

%timeit torch.LongTensor(simple_list)
# 100 loops, best of 5: 0.27 ms per loop

%timeit python_padding(simple_list)
# 100 loops, best of 5: 0.27 ms per loop

%timeit foldedtensor.as_folded_tensor(simple_list)
# 100 loops, best of 5: 0.08 ms per loop

```


## Case 4 (same lengths nested lists to flat tensor)

```python
nested_list = make_nested_list(32, 100, 30, value=1)

%timeit torch.tensor(nested_list).view(-1)
# 100 loops, best of 5: 6.67 ms per loop

%timeit torch.LongTensor(nested_list).view(-1)
# 100 loops, best of 5: 2.74 ms per loop

%timeit python_padding(nested_list).view(-1)
# 100 loops, best of 5: 17.16 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list).view(-1)
# 100 loops, best of 5: 1.02 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list, data_dims=(2,))
# 100 loops, best of 5: 0.95 ms per loop

```

## Case 5 (variable lengths nested lists) to padded embeddings

Nested lists with different lengths (second level lists have lengths between 50 and 150). We compare `foldedtensor` with `torch.nested`.
```python
nested_list = make_nested_list(32, (50, 150), 30, value=1)

# Padding with 0

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 3.11 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list).as_tensor()
# 100 loops, best of 5: 0.90 ms per loop

# Padding with 1

%timeit torch.nested.nested_tensor([torch.FloatTensor(sub) for sub in nested_list]).to_padded_tensor(1)
# 100 loops, best of 5: 3.57 ms per loop

%timeit x = foldedtensor.as_folded_tensor(nested_list); x.masked_fill_(x.mask, 1)
# 100 loops, best of 5: 1.33 ms per loop

```


## Case 6 (2d padding)

```python
nested_list = make_nested_list(160, (50, 150), value=1)

%timeit python_padding(nested_list)
# 100 loops, best of 5: 1.24 ms per loop

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 1.09 ms per loop

%timeit torch.nn.utils.rnn.pad_sequence([torch.LongTensor(sub) for sub in nested_list], batch_first=True, padding_value=0)
# 100 loops, best of 5: 0.78 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 0.13 ms per loop

```
