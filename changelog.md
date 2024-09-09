# Changelog

## v0.3.5

- Support hashing the `folded_tensor.length` field (via a UserList), which is convenient for caching
- Improve error messaging when refolding with missing dims

## v0.3.4

- Fix a data_dims access issue
- Marginally improve the speed of handling FoldedTensors in standard torch operations
- Use default torch types (e.g. `torch.float32` or `torch.torch64`)

## v0.3.3

- Handle empty inputs (e.g. `as_folded_tensor([[[], []], [[]]])`) by returning an empty tensor
- Correctly bubble errors when converting inputs with varying deepness (e.g. `as_folded_tensor([1, [2, 3]])`)

## v0.3.2

- Allow to use `as_folded_tensor` with no args, as a simple padding function

## v0.3.1

- Enable sharing FoldedTensor instances in a multiprocessing + cuda context by autocloning the indexer before fork-pickling an instance
- Distribute arm64 wheels for macOS

## v0.3.0

- Allow dims after last foldable dim during list conversion (e.g. embeddings)

## v0.2.2

- Github release :octocat:
- Fix backpropagation when refolding

## v0.2.1

- Improve performance by computing the new "padded to flattened" indexer only (and not the previous one) when refolding

## v0.2.0

- Remove C++ torch dependency in favor of Numpy due to lack of torch ABI backward/forward compatibility, making the pre-built wheels unusable in most cases
- Require dtype to be specified when creating a FoldedTensor from a nested list

## v0.1.0

Inception ! :tada:

- Support for arbitrary numbers of nested dimensions
- No computational overhead when dealing with already padded tensors
- Dynamic re-padding (or refolding) of data based on stored inner lengths
- Automatic mask generation and updating whenever the tensor is refolded
- C++ optimized code for fast data loading from Python lists and refolding
- Flexibility in data representation, making it easy to switch between different layouts when needed
