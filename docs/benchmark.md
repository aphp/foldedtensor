
Benchmarks
----------

This file was generated from [`scripts/benchmark.py`](../scripts/benchmark.py).

It compares the performance of `foldedtensor` with various alternatives for padding
and working with nested lists and tensors.

Environment:
- `torch.__version__ == '2.6.0'`
- `foldedtensor.__version__ == '0.4.0'`
- `python == 3.9.20`
- `sys.platform == 'darwin'`


## Case 1 (pad variable lengths nested list)

The following 3-levelled nested lists has lengths of 32, then between 50 and 100, and then between 25 and 30.
```python
nested_list = make_nested_list(32, (50, 100), (25, 30), value=1)

Comparisons:
%timeit python_padding(nested_list)
# 100 loops, best of 5: 15.09 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 0.73 ms per loop

```
Speedup against best alternative: **20.67x** :rocket:

## Case 2 (same lengths nested lists)

```python
nested_list = make_nested_list(32, 100, 30, value=1)

%timeit torch.tensor(nested_list)
# 100 loops, best of 5: 6.51 ms per loop

%timeit torch.LongTensor(nested_list)
# 100 loops, best of 5: 2.78 ms per loop

%timeit python_padding(nested_list)
# 100 loops, best of 5: 18.38 ms per loop

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 3.00 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 1.08 ms per loop

```
Speedup against best alternative: **2.58x** :rocket:

## Case 3 (simple list)

```python
simple_list = make_nested_list(10000, value=1)

%timeit torch.tensor(simple_list)
# 100 loops, best of 5: 0.63 ms per loop

%timeit torch.LongTensor(simple_list)
# 100 loops, best of 5: 0.27 ms per loop

%timeit python_padding(simple_list)
# 100 loops, best of 5: 0.28 ms per loop

%timeit foldedtensor.as_folded_tensor(simple_list)
# 100 loops, best of 5: 0.08 ms per loop

```
Speedup against best alternative: **3.32x** :rocket:

## Case 4 (same lengths nested lists to flat tensor)

```python
nested_list = make_nested_list(32, 100, 30, value=1)

%timeit torch.tensor(nested_list).view(-1)
# 100 loops, best of 5: 6.52 ms per loop

%timeit torch.LongTensor(nested_list).view(-1)
# 100 loops, best of 5: 2.76 ms per loop

%timeit python_padding(nested_list).view(-1)
# 100 loops, best of 5: 18.62 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list).view(-1)
# 100 loops, best of 5: 1.12 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list, data_dims=(2,))
# 100 loops, best of 5: 1.08 ms per loop

```
Speedup against best alternative: **2.47x** :rocket:
## Case 5 (variable lengths nested lists) to padded embeddings

Nested lists with different lengths (second level lists have lengths between 50 and 150). We compare `foldedtensor` with `torch.nested`.
```python
nested_list = make_nested_list(32, (50, 150), 30, value=1)

# Padding with 0

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 3.02 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list).as_tensor()
# 100 loops, best of 5: 1.03 ms per loop

```
Speedup against best alternative: **2.95x** :rocket:
```python
# Padding with 1

%timeit torch.nested.nested_tensor([torch.FloatTensor(sub) for sub in nested_list]).to_padded_tensor(1)
# 100 loops, best of 5: 3.72 ms per loop

%timeit x = foldedtensor.as_folded_tensor(nested_list); x.masked_fill_(x.mask, 1)
# 100 loops, best of 5: 1.62 ms per loop

```
Speedup against best alternative: **2.30x** :rocket:

## Case 6 (2d padding)

```python
nested_list = make_nested_list(160, (50, 150), value=1)

%timeit python_padding(nested_list)
# 100 loops, best of 5: 1.33 ms per loop

%timeit torch.nested.nested_tensor([torch.LongTensor(sub) for sub in nested_list]).to_padded_tensor(0)
# 100 loops, best of 5: 1.14 ms per loop

%timeit torch.nn.utils.rnn.pad_sequence([torch.LongTensor(sub) for sub in nested_list], batch_first=True, padding_value=0)
# 100 loops, best of 5: 0.86 ms per loop

%timeit foldedtensor.as_folded_tensor(nested_list)
# 100 loops, best of 5: 0.15 ms per loop

```
Speedup against best alternative: **5.88x** :rocket:
