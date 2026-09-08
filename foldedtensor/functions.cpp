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
        const std::vector<std::vector<int64_t>> &lengths,
        const std::vector<int> &new_data_dims) {
    if (new_data_dims.empty() || lengths.empty() || new_data_dims.back() != lengths.size() - 1) {
        throw py::value_error("data_dims must end with the last variable dimension");
    }
    int previous = -1;
    for (int dim : new_data_dims) {
        if (dim <= previous || dim >= lengths.size()) {
            throw py::value_error("data_dims must be strictly increasing dimension indices within lengths");
        }
        previous = dim;
    }
    const size_t n_lengths = lengths.size();
    const size_t n_new_dims = new_data_dims.size();

    std::vector<int8_t> new_dim_map(n_lengths, -1);
    for (size_t i = 0; i < n_new_dims; i++) {
        new_dim_map[new_data_dims[i]] = i;
    }

    std::vector<size_t> offsets(n_lengths, 0);
    std::vector<int64_t> new_idx(n_new_dims, 0);
    // Store the padded start coordinates and length of each innermost sequence
    std::vector<std::tuple<std::vector<int64_t>, int64_t>> operations;
    operations.reserve(lengths.back().size());

    long long n_elements = 0;
    std::vector<int64_t> new_shape(n_new_dims, 0);
    // Visit every parent so empty sequences keep their position in padded layouts
    auto visit = [&](auto &&self, size_t dim) -> void {
        const auto length = lengths[dim][offsets[dim]++];
        const auto mapped_dim = new_dim_map[dim];
        if (dim == n_lengths - 1) {
            operations.emplace_back(new_idx, length);
            n_elements += length;
            new_idx.back() += length;
            new_shape.back() = std::max(new_shape.back(), new_idx.back());
            return;
        }
        for (int64_t i = 0; i < length; ++i) {
            if (mapped_dim >= 0) {
                new_shape[mapped_dim] = std::max(new_shape[mapped_dim], new_idx[mapped_dim] + 1);
            }
            self(self, dim + 1);
            if (mapped_dim >= 0) {
                ++new_idx[mapped_dim];
                std::fill(new_idx.begin() + mapped_dim + 1, new_idx.end(), 0);
            }
        }
    };
    visit(visit, 0);

    // Row major strides map sequence starts to flat storage positions
    std::vector<long long> new_strides(n_new_dims, 1);
    for (int i = n_new_dims - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    auto new_indexer = py::array_t<int64_t>(n_elements);
    size_t offset = 0;
    for (const auto &operation: operations) {
        const auto &idx = std::get<0>(operation);
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


// Expand range boundaries into storage indices for pooling, excluding padding
std::tuple<py::array, py::array, py::array> make_indices_ranges(
        py::array_t<int64_t, py::array::c_style> begins,
        py::array_t<int64_t, py::array::c_style> ends,
        const std::vector<int> &indice_dims,
        const std::vector<std::vector<int64_t>> &lengths,
        const std::vector<int> &data_dims) {
    if (indice_dims.empty() || begins.ndim() != 2 || ends.ndim() != 2 ||
        begins.shape(0) != indice_dims.size() || ends.shape(0) != indice_dims.size() ||
        begins.shape(1) != ends.shape(1)) {
        throw py::value_error("Coordinate arrays must match indice_dims and the number of ranges");
    }
    int previous = -1;
    for (int dim : indice_dims) {
        if (dim <= previous || dim >= lengths.size()) {
            throw py::value_error("indice_dims must be strictly increasing dimension indices within lengths");
        }
        previous = dim;
    }
    if (data_dims.empty() || data_dims.back() != lengths.size() - 1) {
        throw py::value_error("data_dims must end with the last variable dimension");
    }

    // Prefix offsets map nested coordinates to boundaries in the flattened tensor
    std::vector<std::vector<int64_t>> prefixes(lengths.size());
    for (size_t dim = 0; dim < lengths.size(); ++dim) {
        prefixes[dim].push_back(0);
        for (int64_t length : lengths[dim]) {
            prefixes[dim].push_back(prefixes[dim].back() + length);
        }
    }
    auto flat_boundary = [&](const py::array_t<int64_t, py::array::c_style> &coords, size_t span) {
        auto values = coords.unchecked<2>();
        int dim = indice_dims.front();
        int64_t pos = values(0, span);
        if (pos < 0 || pos > prefixes[dim].back() ||
            (pos == prefixes[dim].back() && indice_dims.size() > 1)) {
            throw py::index_error("Index out of bounds at first dimension");
        }
        for (size_t i = 1; i < indice_dims.size(); ++i) {
            int64_t start = pos, stop = pos + 1;
            for (++dim; dim <= indice_dims[i]; ++dim) {
                start = prefixes[dim].at(start);
                stop = prefixes[dim].at(stop);
            }
            dim = indice_dims[i];
            int64_t coord = values(i, span);
            if (coord < 0 || coord > stop - start ||
                (coord == stop - start && i + 1 < indice_dims.size())) {
                throw py::index_error("Index out of bounds in addressed dimension");
            }
            pos = start + coord;
        }
        for (++dim; dim < lengths.size(); ++dim) {
            pos = prefixes[dim].at(pos);
        }
        return pos;
    };

    const auto count = begins.shape(1);
    std::vector<int64_t> starts(count), stops(count);
    py::array_t<int64_t> offsets(count);
    auto offset = offsets.mutable_unchecked<1>();
    int64_t total = 0;
    for (py::ssize_t i = 0; i < count; ++i) {
        starts[i] = flat_boundary(begins, i);
        stops[i] = flat_boundary(ends, i);
        if (stops[i] < starts[i]) {
            throw py::value_error("Range end before begin");
        }
        offset(i) = total;
        total += stops[i] - starts[i];
    }

    // Map flattened positions to storage rows for padded layouts
    py::array_t<int64_t> indexer;
    if (data_dims.size() > 1) {
        indexer = std::get<0>(make_refolding_indexer(lengths, data_dims));
    }
    py::array_t<int64_t> indices(total), spans(total);
    auto indices_data = indices.mutable_data();
    auto spans_data = spans.mutable_data();
    for (py::ssize_t i = 0; i < count; ++i) {
        for (int64_t pos = starts[i], j = offset(i); pos < stops[i]; ++pos, ++j) {
            indices_data[j] = data_dims.size() == 1 ? pos : indexer.data()[pos];
            spans_data[j] = i;
        }
    }
    return {indices, offsets, spans};
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

PYBIND11_MODULE(_C, m) {
    // Initialize the NumPy API.
    init_numpy();

    m.def("make_indices_ranges", &make_indices_ranges, "Expand ranges into storage indices excluding padding");
    m.def("make_refolding_indexer", &make_refolding_indexer, "Build an indexer to refold data into a different shape");
    m.def("nested_py_list_to_padded_array", &nested_py_list_to_padded_np_array, "Converts a nested Python list to a padded array");
}

#pragma clang diagnostic pop
