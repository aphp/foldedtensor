#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::tuple<
        py::array,          // new indexer
        std::vector<int64_t>// new shape
        >
make_refolding_indexer(
        std::vector<std::vector<int64_t>> &lengths,
        std::vector<int> &new_data_dims) {
    const size_t n_lengths = lengths.size();
    const size_t n_new_dims = new_data_dims.size();

    std::vector<int8_t> new_dim_map(n_lengths, -1);
    for (size_t i = 0; i < n_new_dims; i++) {
        new_dim_map[new_data_dims[i]] = i;
    }

    std::vector<size_t> offsets(n_lengths - 1, 0);
    std::vector<int64_t> new_idx(n_new_dims, 0);
    // Operations are tuples of (indexer, length) that we will use as follows:
    // new_indexer[offset:offset + length] = flat_index(indexer) + range(length)
    std::vector<std::tuple<std::vector<int64_t>, int64_t>> operations;
    operations.reserve(lengths.back().size());

    long long n_elements = 0;
    std::vector<int64_t> new_shape(n_new_dims, 0);
    for (int length: lengths.back()) {
        operations.emplace_back(new_idx, length);

        n_elements += length;

        new_idx.back() += length;
        new_shape[n_new_dims - 1] = std::max(new_shape[n_new_dims - 1], new_idx.back());

        int dim = n_lengths - 2;
        int8_t new_mapped_dim = new_dim_map[dim];
        if (new_mapped_dim >= 0) {
            new_idx[new_mapped_dim] += 1;
            new_shape[new_mapped_dim] = std::max(new_shape[new_mapped_dim], new_idx[new_mapped_dim]);
            for (size_t i = new_mapped_dim + 1; i < n_new_dims; i++) {
                new_idx[i] = 0;
            }
        }

        for (dim = n_lengths - 2; dim >= 0; dim--) {
            lengths[dim][offsets[dim]] -= 1;
            if (lengths[dim][offsets[dim]] > 0) {
                break;
            }

            offsets[dim] += 1;

            if (dim == 0) {
                break;
            }

            int next_dim = dim - 1;
            int8_t next_new_data_mapped_dim = new_dim_map[next_dim];
            if (next_new_data_mapped_dim >= 0) {
                new_idx[next_new_data_mapped_dim] += 1;
                new_shape[next_new_data_mapped_dim] = std::max(new_shape[next_new_data_mapped_dim], new_idx[next_new_data_mapped_dim]);

                for (int8_t i = next_new_data_mapped_dim + 1; i < n_new_dims; i++) {
                    new_idx[i] = 0;
                }
            }
        }
    }
    // Init new strides (full of 1, size = n_old_dims) and compute them in reverse
    // for data size and new data sizes from n_new_dims and n_old_dims offsets
    std::vector<long long> new_strides(n_new_dims, 1);
    for (int i = n_new_dims - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    auto new_indexer = py::array_t<int64_t>(n_elements);
    size_t offset = 0;
    for (auto operation: operations) {
        std::vector<int64_t> &idx = std::get<0>(operation);
        auto length = std::get<1>(operation);

        int64_t begin_idx = 0;
        for (size_t i = 0; i < n_new_dims - 1; ++i) {
            begin_idx += idx[i] * new_strides[i];
        }
        begin_idx += idx.back();

        auto *new_indexer_data = (int64_t *) new_indexer.mutable_data() + offset;
        for (int64_t i = 0; i < length; ++i) {
            *(new_indexer_data + i) = begin_idx + i;
        }
        offset += length;
    }

    return {new_indexer, new_shape};
}


#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

size_t flatten_py_list(
        py::list nested_list,
        std::vector<std::vector<int64_t>> &lengths,
        std::vector<std::tuple<std::vector<int64_t>, int64_t, PyObject *>> &operations,
        std::vector<int64_t> &current_indices,
        std::vector<int64_t> &data_dim_map,
        std::vector<int64_t> &shape,
        int dim,
        int total) {
    if (nested_list.empty()) {
        return total;
    }

    bool is_data_dim = dim >= data_dim_map.size() || data_dim_map[dim] >= 0;
    bool next_is_foldable_dim = dim + 1 < data_dim_map.size();

    if (!py::isinstance<py::list>(nested_list[0])) {
        if (dim < data_dim_map.size() - 1) {
            throw py::value_error(
                    "The provided data_dims have too many entries compared "
                    "to the nesting of the input."
            );
        }
        operations.emplace_back(
                current_indices,
                total,
                PySequence_Fast(nested_list.ptr(), "Something when wrong when reading one of the inner lists"));
        if (dim + 1 == data_dim_map.size()) {
            total += nested_list.size();
        }
        current_indices.back() += nested_list.size();
    } else {
        if (next_is_foldable_dim && lengths.size() <= dim + 1) {
            lengths.emplace_back();
        }

        for (auto &&i: nested_list) {

            auto sublist = i.cast<py::list>();

            if (next_is_foldable_dim) {
                lengths[dim + 1].push_back(sublist.size());
            }

            if (is_data_dim) {
                current_indices.push_back(0);
            }

            total = flatten_py_list(
                    sublist,
                    lengths,
                    operations,
                    current_indices,
                    data_dim_map,
                    shape,
                    dim + 1,
                    total);

            if (current_indices.size() > shape.size()) {
                shape.push_back(0);
            }

            shape[current_indices.size() - 1] = std::max(shape[current_indices.size() - 1], current_indices.back());

            if (is_data_dim) {

                current_indices.pop_back();
                current_indices.back() += 1;
            }

            if (dim + 1 == data_dim_map.size()) {
                total += 1;
            }
        }
    }

    shape[current_indices.size() - 1] = std::max(shape[current_indices.size() - 1], current_indices.back());

    return total;
}

#pragma clang diagnostic pop

std::tuple<
        py::array,                        // padded array
        py::array,                        // indexer
        std::vector<std::vector<int64_t>>>// lengths
nested_py_list_to_padded_np_array(
        const py::list &nested_list,
        std::vector<int> data_dims,
        py::dtype &dtype) {
    // Will contain the variable lengths of the nested lists
    // One sequence per dimension, containing the lengths of the lists at that dimension
    std::vector<std::vector<int64_t>> lengths;

    // Operations to perform to assign the values to the padded array
    // Each operation is a tuple of the indices of the first element to assign
    // and a list of values to assign contiguously from that position
    std::vector<std::tuple<std::vector<int64_t>, int64_t, PyObject *>> operations;

    // Index from the data dimension to the dim in the theoretically fully padded list
    int64_t max_depth = 0;
    if (data_dims.size() > 0) {
        max_depth = *std::max_element(data_dims.begin(), data_dims.end()) + 1;
    }
    std::vector<int64_t> data_dim_map(max_depth, -1);
    for (unsigned long i = 0; i < data_dims.size(); i++) {
        data_dim_map[data_dims[i]] = i;
    }
    lengths.emplace_back(1, nested_list.size());
    // Shape of the array, will be updated during `flatten_py_list`
    std::vector<int64_t> shape(data_dims.size(), 0);
    // Current indices in the nested list, will be updated during `flatten_py_list`
    std::vector<int64_t> current_indices(1, 0);
    size_t num_elements = flatten_py_list(
            nested_list,
            lengths,
            operations,
            current_indices,
            data_dim_map,
            shape,
            0,
            0);

    // Create the padded array from the shape inferred during `flatten_py_list`
    py::array padded_array = py::array(py::dtype(dtype), shape);
    padded_array[py::make_tuple(py::ellipsis())] = 0;

    // Get the strides of the array
    const py::ssize_t *array_strides = padded_array.strides();
    const size_t itemsize = padded_array.itemsize();

    size_t element_size = 1;
    for (int i = data_dims.size(); i < shape.size(); ++i) {
        element_size *= shape[i];
    }
    py::array indexer = py::array(py::dtype::of<int64_t>(), num_elements);
    for (const auto &op: operations) {
        auto indices = std::get<0>(op);
        auto element_idx = std::get<1>(op);
        auto flat_list = std::get<2>(op);

        // Byte pointers to elements in the array and the mask
        // Since we cannot know the size of elements in the array (as it's handled
        // dynamically by numpy), we use byte pointers to move around in `padded_array`
        // and will increase the pointer by `itemsize` to move from one element
        // to the next.

        // This is the byte index of the first element set by the current operation
        // Once divided by the number of bytes per element, this will be used to fill the
        // indexer (which can be used to get the elements position in the padded array).
        size_t begin_byte = 0;
        for (size_t dim = 0; dim < indices.size(); ++dim) {
            // Since mask is boolean, every element is 1 byte, therefore
            // we can use strides to count the number of elements
            begin_byte += array_strides[dim] * indices[dim];
        }
        size_t begin_idx = begin_byte / (itemsize * element_size);
        auto *array_ptr = (char *) padded_array.mutable_data() + begin_byte;
        int64_t *indexer_ptr;
        if (element_size > 1) {
            *(((int64_t *) indexer.mutable_data()) + element_idx) = begin_idx;
        } else {
            indexer_ptr = ((int64_t *) indexer.mutable_data()) + element_idx;
        }

        PyObject **items = PySequence_Fast_ITEMS(flat_list);
        const int length = PySequence_Fast_GET_SIZE(flat_list);
        for (int i = 0; i < length; ++i) {
            // Set the element in the array and move to the next element
            // Since array elements can be of any size, we use the element size (in
            // bytes) to move from one element to the next
            if (PyArray_SETITEM((PyArrayObject *)padded_array.ptr(), array_ptr, items[i]) < 0) {
                throw py::error_already_set();
            }
            array_ptr += itemsize;

            // Assign the current index to the indexer and move to the next element
            if (element_size == 1) {
                *indexer_ptr = begin_idx + i;
                indexer_ptr += 1;
            }
        }

        Py_DECREF(flat_list);
    }

    return {padded_array, indexer, lengths};
}

// Helper function to initialize the NumPy C API.
static bool init_numpy() {
    import_array();
    return true;
}

static std::vector<int64_t> cumsum(const std::vector<int64_t> &v) {
    std::vector<int64_t> out;
    out.reserve(v.size() + 1);
    out.push_back(0);
    int64_t total = 0;
    for (auto x : v) {
        total += x;
        out.push_back(total);
    }
    return out;
}

/**
 * Compute per-dimension child start offsets (exclusive prefix sums).
 *
 * For every variable dimension j>0, `lengths[j]` contains, for each parent
 * entity at dimension j-1, the number of children at dimension j. The
 * exclusive prefix-sum of this array maps a parent global id to the global id
 * of its first child at the next dimension.
 *
 * - starts[j].size() == lengths[j].size() + 1
 * - For a parent global id g at dimension j-1, the first child global id at
 *   dimension j is `starts[j][g]`, and the number of children is
 *   `lengths[j][g]`.
 * - starts[0] is left empty (unused), since there is no dimension -1.
 *
 * @param lengths Variable lengths per dimension. For j>0, lengths[j][g] is the
 *                number of children in dim j for parent g in dim j-1.
 * @return For each j, starts[j] = cumsum(lengths[j]) (exclusive prefix sum).
 */
static std::vector<std::vector<int64_t>> child_start_offsets(
        const std::vector<std::vector<int64_t>> &lengths) {
    const size_t D = lengths.size();
    std::vector<std::vector<int64_t>> starts(D);
    for (size_t j = 1; j < D; ++j) {
        starts[j] = cumsum(lengths[j]);
    }
    return starts;
}

static std::vector<std::vector<int64_t>> leaf_offsets_per_dim(
    const std::vector<std::vector<int64_t>> &lengths
) {
    const size_t D = lengths.size();
    if (D < 2) {
        return std::vector<std::vector<int64_t>>();
    }
    // Start with one leaf per word
    size_t n_words = 0;
    for (auto x : lengths[D - 2]) n_words += x;
    std::vector<int64_t> counts(n_words, 1);

    std::vector<std::vector<int64_t>> offsets(D);
    // For words (D-2): [0,1,2,...,n_words]
    offsets[D - 2].resize(n_words + 1);
    for (size_t i = 0; i < n_words + 1; ++i) offsets[D - 2][i] = (int64_t)i;

    for (int d = (int)D - 3; d >= 0; --d) {
        std::vector<int64_t> new_counts;
        new_counts.reserve(lengths[d + 1].size());
        auto it = counts.begin();
        for (auto n_children : lengths[d + 1]) {
            int64_t s = 0;
            for (int64_t k = 0; k < n_children; ++k) {
                if (it == counts.end()) break;
                s += *it;
                ++it;
            }
            new_counts.push_back(s);
        }
        counts.swap(new_counts);
        offsets[d] = cumsum(counts);
    }
    return offsets;
}

/**
 * Compute the flat begin index for every leaf under a refolded layout.
 *
 * Given the nested `lengths` description and the list of data dimensions
 * `data_dims` (which must end at the leaf dimension D-1), this function
 * simulates iterating leaves (tokens) while incrementing the multi-dimensional
 * index over the data layout. It returns, for each leaf (global leaf id), the
 * flat index at which that leaf begins in the contiguous, refolded array.
 *
 * The resulting flat indices are computed using strides derived from the
 * maximum extents observed during the simulated iteration of `data_dims`.
 *
 * @param lengths Variable lengths per dimension
 * @param data_dims Contiguous data dimensions in order, must end at D-1.
 * @return Vector `begins[leaf_gid]` giving the flat begin offset of each leaf.
 * @throws std::invalid_argument if `data_dims` does not end with D-1.
 */
static std::vector<int64_t> begin_idx_per_leaf(
        std::vector<std::vector<int64_t>> lengths,
        const std::vector<int> &data_dims) {
    const size_t D = lengths.size();
    const size_t n_new = data_dims.size();
    if (n_new == 0) return {};
    if ((size_t)data_dims.back() != D - 1) {
        throw std::invalid_argument("data_dims must end with last variable dimension");
    }

    std::vector<int8_t> new_dim_map(D, -1);
    for (size_t i = 0; i < n_new; ++i) new_dim_map[data_dims[i]] = (int8_t)i;

    std::vector<int64_t> new_idx(n_new, 0);
    std::vector<int64_t> new_shape(n_new, 0);
    std::vector<size_t> offsets(D - 1, 0);

    std::vector<std::pair<std::vector<int64_t>, int64_t>> ops; // (idx snapshot, leaf length)
    ops.reserve(lengths.back().size());

    for (auto leaf_len : lengths.back()) {
        ops.emplace_back(new_idx, leaf_len);

        new_idx.back() += leaf_len;
        if (new_idx.back() > new_shape.back()) new_shape.back() = new_idx.back();

        int dim = (int)D - 2;
        int8_t mapped = new_dim_map[dim];
        if (mapped >= 0) {
            new_idx[mapped] += 1;
            if (new_idx[mapped] > new_shape[mapped]) new_shape[mapped] = new_idx[mapped];
            for (size_t i = mapped + 1; i < n_new; ++i) new_idx[i] = 0;
        }

        for (dim = (int)D - 2; dim >= 0; --dim) {
            lengths[dim][offsets[dim]] -= 1;
            if (lengths[dim][offsets[dim]] > 0) {
                break;
            }
            offsets[dim] += 1;
            if (dim == 0) break;
            int next_dim = dim - 1;
            int8_t next_mapped = new_dim_map[next_dim];
            if (next_mapped >= 0) {
                new_idx[next_mapped] += 1;
                if (new_idx[next_mapped] > new_shape[next_mapped]) new_shape[next_mapped] = new_idx[next_mapped];
                for (int8_t i = next_mapped + 1; i < (int8_t)n_new; ++i) new_idx[i] = 0;
            }
        }
    }

    // strides
    std::vector<int64_t> strides(n_new, 1);
    for (int i = (int)n_new - 2; i >= 0; --i) {
        int64_t s = new_shape[i + 1];
        if (s <= 0) s = 1;
        strides[i] = strides[i + 1] * s;
    }

    std::vector<int64_t> begins;
    begins.reserve(ops.size());
    for (auto &op : ops) {
        auto &idx = op.first;
        int64_t base = 0;
        for (size_t i = 0; i + 1 < n_new; ++i) base += idx[i] * strides[i];
        base += idx.back();
        begins.push_back(base);
    }
    return begins;
}

/**
 * Resolve a (possibly multi-dimensional) coordinate into a flat token index.
 *
 * The coordinate spans the contiguous variable dimensions given by
 * `indice_dims`. Depending on the last addressed dimension, the function
 * supports boundary indices (equal to the size) and maps them to the logical
 * end position after the last token of the addressed entity/leaf.
 *
 * Single-dimension addressing rules:
 * - If d == D-1 (token dimension): idx in [0, total_tokens] -> begin_of_leaf + offset.
 * - If d == D-2 (leaf/word id): idx in [0, total_words] -> begin_of_leaf.
 * - Else (higher level): idx in [0, leaf_offs[d].size()-1] -> first token of entity.
 *   In all cases, idx == size selects the end position after the last token.
 *
 * Multi-dimension addressing (contiguous): interpret `coord` as offsets within
 * the subtree rooted at `indice_dims[0]`, descend using `starts` to compute the
 * parent global id, and resolve the last coordinate either to a token offset or
 * to the first token of the targeted child and boundary at the last dimension is
 * supported analogously.
 *
 * @param lengths Variable lengths per dimension.
 * @param data_dims Data dimensions (must end at D-1).
 * @param indice_dims Contiguous addressed variable dimensions.
 * @param starts Per-dimension child start offsets: starts[j] = cumsum(lengths[j]).
 * @param leaf_offs For each dimension, offsets into the leaf (token) axis.
 * @param begins_per_leaf Flat begin index per leaf (from begin_idx_per_leaf).
 * @param token_starts Global token cumsum across leaves.
 * @param coord Coordinate values aligned with `indice_dims`.
 * @return Flat token index (or end position) in the refolded layout.
 * @throws std::out_of_range on invalid coordinates beyond the allowed boundary.
 */
// Helper: memoized count of descendants at a target dimension under an entity.
// cache[target_dim][from_dim] is a vector of size = number of entities at from_dim,
// storing the count of target_dim entities under each entity at from_dim.
static int64_t count_descendants_memo(
        const std::vector<std::vector<int64_t>> &lengths,
        const std::vector<std::vector<int64_t>> &starts,
        int from_dim,
        int64_t gid,
        int target_dim,
        std::vector<std::vector<std::vector<int64_t>>> &cache) {
    if (from_dim == target_dim) return 1; // the entity itself counts as 1 at its own dimension
    auto &level_cache = cache[target_dim][from_dim];
    if (gid < 0 || gid >= (int64_t)level_cache.size()) return 0;
    int64_t val = level_cache[gid];
    if (val >= 0) return val;
    // Sum descendant counts over immediate children
    int next_dim = from_dim + 1;
    int64_t n_children = lengths[next_dim][gid];
    int64_t start = starts[next_dim][gid];
    int64_t total = 0;
    for (int64_t i = 0; i < n_children; ++i) {
        total += count_descendants_memo(lengths, starts, next_dim, start + i, target_dim, cache);
    }
    level_cache[gid] = total;
    return total;
}

// Map a flattened offset within descendants at target_dim to a concrete child gid at target_dim.
static int64_t descendant_gid_by_flat_offset(
        const std::vector<std::vector<int64_t>> &lengths,
        const std::vector<std::vector<int64_t>> &starts,
        int from_dim,
        int64_t gid,
        int target_dim,
        int64_t offset,
        std::vector<std::vector<std::vector<int64_t>>> &cache) {
    if (from_dim == target_dim) return gid;
    int next_dim = from_dim + 1;
    int64_t n_children = lengths[next_dim][gid];
    int64_t start = starts[next_dim][gid];
    for (int64_t i = 0; i < n_children; ++i) {
        int64_t child_gid = start + i;
        int64_t cnt = count_descendants_memo(lengths, starts, next_dim, child_gid, target_dim, cache);
        if (offset < cnt) {
            return descendant_gid_by_flat_offset(lengths, starts, next_dim, child_gid, target_dim, offset, cache);
        }
        offset -= cnt;
    }
    // Should not reach here if offset < total descendants
    throw std::out_of_range("Offset beyond descendant count");
}

static int64_t compute_flat_index(
        const std::vector<std::vector<int64_t>> &lengths,
        const std::vector<int> &data_dims,
        const std::vector<int> &indice_dims,
        const std::vector<std::vector<int64_t>> &starts,
        const std::vector<std::vector<int64_t>> &leaf_offs,
        const std::vector<int64_t> &begins_per_leaf,
        const std::vector<int64_t> &token_starts,
        const std::vector<int64_t> &coord) {
    const size_t D = lengths.size();
    const int dend = indice_dims.back();

    // Single-dimension convenience addressing
    if (indice_dims.size() == 1) {
        const int d = indice_dims[0];
        int64_t idx = coord[0];
        if (d == (int)D - 1) {
            // Global token index (pooled across leaves), boundary allowed
            int64_t total_tokens = token_starts.back();
            if (idx < 0 || idx > total_tokens) throw std::out_of_range("Token index out of bounds");
            if (idx == total_tokens) {
                int64_t last_leaf = (int64_t)lengths.back().size() - 1;
                return begins_per_leaf[last_leaf] + lengths.back()[last_leaf];
            }
            auto it = std::upper_bound(token_starts.begin(), token_starts.end(), idx);
            int64_t leaf = (int64_t)(it - token_starts.begin()) - 1;
            int64_t offset = idx - token_starts[leaf];
            return begins_per_leaf[leaf] + offset;
        } else if (d == (int)D - 2) {
            // Global word index (leaf id), boundary allowed
            int64_t total_words = 0; for (auto x : lengths[D - 2]) total_words += x;
            if (idx < 0 || idx > total_words) throw std::out_of_range("Word index out of bounds");
            if (idx == total_words) {
                int64_t last_leaf = total_words - 1;
                return begins_per_leaf[last_leaf] + lengths.back()[last_leaf];
            }
            return begins_per_leaf[idx];
        } else {
            // Higher-level entity: map to first token of its first leaf, boundary allowed
            // Bounds are based on the total number of entities at this level across parents,
            // which corresponds to leaf_offs[d].size() - 1, not lengths[d].size().
            int64_t total_entities = (int64_t)leaf_offs[d].size() - 1;
            if (idx < 0 || idx > total_entities) throw std::out_of_range("Index out of bounds");
            if (idx == total_entities) {
                int64_t leaf_end = leaf_offs[d].back();
                if (leaf_end == 0) return 0; // empty
                int64_t last_leaf = leaf_end - 1;
                return begins_per_leaf[last_leaf] + lengths.back()[last_leaf];
            }
            int64_t leaf_idx = leaf_offs[d][idx];
            return begins_per_leaf[leaf_idx];
        }
    }

    // General non-contiguous multi-dimension addressing (flatten intermediate dims)
    // Validate strictly increasing dims
    for (size_t i = 1; i < indice_dims.size(); ++i) {
        if (indice_dims[i] <= indice_dims[i - 1]) {
            throw std::invalid_argument("indice_dims must be strictly increasing");
        }
    }

    // Build a cache for descendant counts for all target dims that may be addressed
    // Prepare cache[target_dim][from_dim][gid] = count, initialized to -1
    std::vector<std::vector<std::vector<int64_t>>> cache;
    cache.resize(D);
    for (size_t t = 0; t < D; ++t) {
        cache[t].resize(D);
        for (size_t fd = 0; fd < D; ++fd) {
            size_t n_entities = 0;
            if (fd == D - 1) {
                n_entities = lengths.back().size();
            } else if (fd + 1 < D) {
                n_entities = lengths[fd + 1].size();
            }
            cache[t][fd] = std::vector<int64_t>(n_entities, -1);
        }
    }

    // Resolve the first coordinate to a global entity id at its dim
    int d0 = indice_dims[0];
    int64_t gid = coord[0];
    // Bounds for first coordinate (no boundary allowed except if last dim only, already handled)
    if (d0 == (int)D - 1) {
        throw std::invalid_argument("First indice_dim cannot be the leaf/token dimension when multiple dims are provided");
    } else if (d0 == (int)D - 2) {
        int64_t total_words = 0; for (auto x : lengths[D - 2]) total_words += x;
        if (gid < 0 || gid >= total_words) throw std::out_of_range("Index out of bounds at first dimension");
    } else {
        int64_t total_entities = (int64_t)leaf_offs[d0].size() - 1;
        if (gid < 0 || gid >= total_entities) throw std::out_of_range("Index out of bounds at first dimension");
    }

    if (indice_dims.size() == 2 && dend == (int)D - 1) {
        // Common case: [d_parent, token] with flattening across intermediates
        int parent_dim = d0;
        int64_t last_idx = coord[1];
        // Number of tokens under this parent entity
        int64_t token_count = count_descendants_memo(lengths, starts, parent_dim, gid, (int)D - 1, cache);
        if (last_idx < 0 || last_idx > token_count) throw std::out_of_range("Token index out of bounds");
        int64_t leaf_begin = leaf_offs[parent_dim][gid];
        int64_t leaf_end = leaf_offs[parent_dim][gid + 1];
        if (last_idx == token_count) {
            if (leaf_end == leaf_begin) return begins_per_leaf[leaf_begin];
            int64_t last_leaf = leaf_end - 1;
            return begins_per_leaf[last_leaf] + lengths.back()[last_leaf];
        }
        // Map token offset to absolute token index across global token_starts
        int64_t base_tokens = token_starts[leaf_begin];
        int64_t abs_token = base_tokens + last_idx;
        auto it = std::upper_bound(token_starts.begin(), token_starts.end(), abs_token);
        int64_t leaf = (int64_t)(it - token_starts.begin()) - 1;
        int64_t offset = abs_token - token_starts[leaf];
        return begins_per_leaf[leaf] + offset;
    }

    // Traverse successive addressed dims, skipping/flattening intermediates
    for (size_t i = 1; i + 1 < indice_dims.size(); ++i) {
        int target_dim = indice_dims[i];
        int64_t off = coord[i];
        if (off < 0) throw std::out_of_range("Negative index not allowed");
        int64_t cnt = count_descendants_memo(lengths, starts, indice_dims[i - 1], gid, target_dim, cache);
        if (off >= cnt) throw std::out_of_range("Index out of bounds at intermediate dimension");
        gid = descendant_gid_by_flat_offset(lengths, starts, indice_dims[i - 1], gid, target_dim, off, cache);
    }

    // Handle last dimension
    int prev_dim = indice_dims[indice_dims.size() - 2];
    int64_t last_idx = coord.back();
    if (dend == (int)D - 1) {
        // last is token, gid is entity at prev_dim
        int64_t token_count = count_descendants_memo(lengths, starts, prev_dim, gid, (int)D - 1, cache);
        if (last_idx < 0 || last_idx > token_count) throw std::out_of_range("Token index out of bounds");
        int64_t leaf_begin = leaf_offs[prev_dim][gid];
        int64_t leaf_end = leaf_offs[prev_dim][gid + 1];
        if (last_idx == token_count) {
            if (leaf_end == leaf_begin) return begins_per_leaf[leaf_begin];
            int64_t last_leaf = leaf_end - 1;
            return begins_per_leaf[last_leaf] + lengths.back()[last_leaf];
        }
        int64_t base_tokens = token_starts[leaf_begin];
        int64_t abs_token = base_tokens + last_idx;
        auto it = std::upper_bound(token_starts.begin(), token_starts.end(), abs_token);
        int64_t leaf = (int64_t)(it - token_starts.begin()) - 1;
        int64_t offset = abs_token - token_starts[leaf];
        return begins_per_leaf[leaf] + offset;
    } else {
        // last is an addressed non-leaf level, select descendant at dend with boundary allowed
        int64_t cnt = count_descendants_memo(lengths, starts, prev_dim, gid, dend, cache);
        if (last_idx < 0 || last_idx > cnt) throw std::out_of_range("Index out of bounds at last dimension");
        if (last_idx == cnt) {
            int64_t leaf_begin = leaf_offs[prev_dim][gid];
            int64_t leaf_end = leaf_offs[prev_dim][gid + 1];
            if (leaf_end == leaf_begin) return begins_per_leaf[leaf_begin];
            int64_t last_leaf = leaf_end - 1;
            return begins_per_leaf[last_leaf] + lengths.back()[last_leaf];
        }
        int64_t child_gid = descendant_gid_by_flat_offset(lengths, starts, prev_dim, gid, dend, last_idx, cache);
        int64_t leaf_idx = (dend == (int)D - 2) ? child_gid : leaf_offs[dend][child_gid];
        return begins_per_leaf[leaf_idx];
    }
}

static py::array_t<int64_t> map_indices_cpp(
        const std::vector<std::vector<int64_t>> &lengths,
        const std::vector<int> &data_dims,
        const std::vector<int> &indice_dims,
        const std::vector<std::vector<int64_t>> &indices) {
    const size_t D = lengths.size();
    if (D == 0) return py::array_t<int64_t>(0);

    if (data_dims.empty() || (size_t)data_dims.back() != D - 1) {
        throw std::invalid_argument("data_dims must end with the last variable dimension");
    }
    if (indices.size() != indice_dims.size()) {
        throw std::invalid_argument("indices and indice_dims must have the same length");
    }
    size_t n = indices.empty() ? 0 : indices[0].size();
    for (auto &v : indices) if (v.size() != n) throw std::invalid_argument("indices must be same length");

    // Precompute helpers
    std::vector<int64_t> begins = begin_idx_per_leaf(lengths, data_dims);
    std::vector<std::vector<int64_t>> leaf_offs = leaf_offsets_per_dim(lengths);
    std::vector<std::vector<int64_t>> starts = child_start_offsets(lengths);
    std::vector<int64_t> token_starts = cumsum(lengths.back());

    // Validate monotonic increasing dims and bounds
    if (!indice_dims.empty()) {
        for (size_t i = 1; i < indice_dims.size(); ++i) {
            if (indice_dims[i] <= indice_dims[i - 1]) {
                throw std::invalid_argument("indice_dims must be strictly increasing");
            }
        }
        if (indice_dims.back() > (int)D - 1) {
            throw std::invalid_argument("Final indice_dim must be <= leaf dimension");
        }
    }

    py::array_t<int64_t> out(n);
    auto *out_ptr = (int64_t *) out.mutable_data();
    std::vector<int64_t> coord(indice_dims.size());
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < indice_dims.size(); ++j) coord[j] = indices[j][i];
        out_ptr[i] = compute_flat_index(lengths, data_dims, indice_dims, starts, leaf_offs, begins, token_starts, coord);
    }
    return out;
}

// Extracted from inline binding: build flat indices for spans between begins and ends
static py::tuple make_indices_ranges_cpp(
        const std::vector<std::vector<int64_t>> &lengths,
        const std::vector<int> &data_dims,
        const std::vector<int> &indice_dims,
        const std::vector<std::vector<int64_t>> &begins,
        const std::vector<std::vector<int64_t>> &ends) {
    const size_t D = lengths.size();
    if (data_dims.empty() || (size_t)data_dims.back() != D - 1) {
        throw std::invalid_argument("data_dims must end with the last variable dimension");
    }
    if (begins.size() != ends.size() || begins.size() != indice_dims.size()) {
        throw std::invalid_argument("begins/ends must match indice_dims length");
    }
    size_t n = begins.empty() ? 0 : begins[0].size();
    for (auto &v : begins) if (v.size() != n) throw std::invalid_argument("begins arrays must be same length");
    for (auto &v : ends) if (v.size() != n) throw std::invalid_argument("ends arrays must be same length as begins");

    // Precompute helpers
    std::vector<int64_t> begins_per_leaf = begin_idx_per_leaf(lengths, data_dims);
    std::vector<std::vector<int64_t>> leaf_offs = leaf_offsets_per_dim(lengths);
    std::vector<std::vector<int64_t>> starts = child_start_offsets(lengths);
    std::vector<int64_t> token_starts = cumsum(lengths.back());

    // Validate monotonic increasing dims
    if (!indice_dims.empty()) {
        for (size_t i = 1; i < indice_dims.size(); ++i) {
            if (indice_dims[i] <= indice_dims[i - 1]) {
                throw std::invalid_argument("indice_dims must be strictly increasing");
            }
        }
        if (indice_dims.back() > (int)D - 1) {
            throw std::invalid_argument("Final indice_dim must be <= leaf dimension");
        }
    }

    // First pass: compute starts and total length
    std::vector<int64_t> starts_vec;
    starts_vec.reserve(n);
    std::vector<std::pair<int64_t, int64_t>> be_pairs;
    be_pairs.reserve(n);
    int64_t total = 0;
    for (size_t i = 0; i < n; ++i) {
        std::vector<int64_t> bcoord(indice_dims.size());
        std::vector<int64_t> ecoord(indice_dims.size());
        for (size_t j = 0; j < indice_dims.size(); ++j) {
            bcoord[j] = begins[j][i];
            ecoord[j] = ends[j][i];
        }
        int64_t b = compute_flat_index(lengths, data_dims, indice_dims, starts, leaf_offs, begins_per_leaf, token_starts, bcoord);
        int64_t e = compute_flat_index(lengths, data_dims, indice_dims, starts, leaf_offs, begins_per_leaf, token_starts, ecoord);
        if (e < b) throw std::invalid_argument("Range end before begin");
        starts_vec.push_back(total);
        be_pairs.emplace_back(b, e);
        total += (e - b);
    }

    // Build outputs
    py::array_t<int64_t> indices(total);
    auto *ind_ptr = (int64_t *) indices.mutable_data();
    // Also build span indices: the span number for each expanded position
    py::array_t<int64_t> span_indices(total);
    auto *span_ptr = (int64_t *) span_indices.mutable_data();
    for (size_t i = 0; i < be_pairs.size(); ++i) {
        auto &p = be_pairs[i];
        for (int64_t x = p.first; x < p.second; ++x) {
            *ind_ptr++ = x;
            *span_ptr++ = (int64_t) i;
        }
    }
    py::array_t<int64_t> offsets(starts_vec.size());
    auto *off_ptr = (int64_t *) offsets.mutable_data();
    for (auto s : starts_vec) *off_ptr++ = s;
    return py::make_tuple(indices, offsets, span_indices);
}

PYBIND11_MODULE(_C, m) {
    // Initialize the NumPy API.
    init_numpy();

    m.def("make_refolding_indexer", &make_refolding_indexer, "Build an indexer to refold data into a different shape");
    m.def("nested_py_list_to_padded_array", &nested_py_list_to_padded_np_array, "Converts a nested Python list to a padded array");
    m.def("map_indices", &map_indices_cpp, "Maps indices to flat leaf starts with boundary support");
    m.def("make_indices_ranges", &make_indices_ranges_cpp, "Expand ranges between begins and ends into flat indices, start offsets, and span indices");
}

// PARTS TO SIMPLIFY -- END

#pragma clang diagnostic pop
