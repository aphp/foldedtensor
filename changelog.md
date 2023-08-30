# v0.3.1

- Enable sharing FoldedTensor instances in a multiprocessing + cuda context by autocloning the indexer before fork-pickling an instance
- Distribute arm64 wheels for macOS

# v0.3.0

- Allow dims after last foldable dim during list conversion (e.g. embeddings)

# v0.2.2

- Github release :octocat:
- Fix backpropagation when refolding

# v0.2.1

- Improve performance by computing the new "padded to flattened" indexer only (and not the previous one) when refolding

# v0.2.0

- Remove C++ torch dependency in favor of Numpy due to lack of torch ABI backward/forward compatibility, making the pre-built wheels unusable in most cases
- Require dtype to be specified when creating a FoldedTensor from a nested list

# v0.1.0

Inception ! :tada:

- Support for arbitrary numbers of nested dimensions
- No computational overhead when dealing with already padded tensors
- Dynamic re-padding (or refolding) of data based on stored inner lengths
- Automatic mask generation and updating whenever the tensor is refolded
- C++ optimized code for fast data loading from Python lists and refolding
- Flexibility in data representation, making it easy to switch between different layouts when needed
