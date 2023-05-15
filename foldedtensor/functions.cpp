#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::tuple<py::array, py::array, std::vector<int64_t>, py::array> make_refolding_indexers(
        // const py::array &data,
        std::vector<std::vector<int64_t>> &lengths,
        std::vector<int> &old_shape,
        std::vector<int> &old_data_dims,
        std::vector<int> &new_data_dims) {
    // torch::NoGradGuard no_grad;

    const size_t n_lengths = lengths.size();
    const size_t n_old_dims = old_data_dims.size();
    const size_t n_new_dims = new_data_dims.size();

    std::vector<int8_t> old_dim_map(n_lengths, -1);
    for (size_t i = 0; i < n_old_dims; i++) {
        old_dim_map[old_data_dims[i]] = i;
    }

    std::vector<int8_t> new_dim_map(n_lengths, -1);
    for (size_t i = 0; i < n_new_dims; i++) {
        new_dim_map[new_data_dims[i]] = i;
    }

    std::vector<size_t> offsets(n_lengths - 1, 0);
    std::vector<int64_t> old_idx(n_old_dims, 0);
    std::vector<int64_t> new_idx(n_new_dims, 0);
    std::vector<std::tuple<
            std::vector<int64_t>,
            std::vector<int64_t>,
            int64_t>>
            operations;
    operations.reserve(lengths.back().size());
    long long n_elements = 0;
    std::vector<int64_t> new_shape(n_new_dims + old_shape.size() - n_old_dims, 0);
    for (size_t i = 0; i < old_shape.size() - n_old_dims; i++) {
        new_shape[n_new_dims + i] = old_shape[n_old_dims + i];
    }
    for (int length: lengths.back()) {
        operations.emplace_back(old_idx, new_idx, length);

        n_elements += length;

        new_idx.back() += length;
        old_idx.back() += length;
        new_shape[n_new_dims - 1] = std::max(new_shape[n_new_dims - 1], new_idx.back());

        int dim = old_dim_map.size() - 2;
        int8_t old_mapped_dim = old_dim_map[dim];
        if (old_mapped_dim >= 0) {
            old_idx[old_mapped_dim] += 1;
            for (size_t i = old_mapped_dim + 1; i < n_old_dims; i++) {
                old_idx[i] = 0;
            }
        }

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
            int8_t next_old_data_mapped_dim = old_dim_map[next_dim];
            if (next_old_data_mapped_dim >= 0) {
                old_idx[next_old_data_mapped_dim] += 1;
                for (int8_t i = next_old_data_mapped_dim + 1; i < n_old_dims; i++) {
                    old_idx[i] = 0;
                }
            }

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


    std::vector<int64_t> data_flat_dims = {-1};

    int64_t new_data_num_padded = 1;
    for (int64_t i = 0; i < n_new_dims; ++i) {
        new_data_num_padded *= new_shape[i];
    }

    auto old_indexer = py::array_t<int64_t>(n_elements);
    auto new_indexer = py::array_t<int64_t>(n_elements);

    auto mask = py::array_t<uint8_t>(new_data_num_padded);
    mask[py::make_tuple(py::ellipsis())] = 0;

    // Init new strides (full of 1, size = n_old_dims) and compute them in reverse
    // for data size and new data sizes from n_new_dims and n_old_dims offsets
    std::vector<long long> old_strides(n_old_dims, 1);
    for (int i = n_old_dims - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }
    std::vector<long long> new_strides(n_new_dims, 1);
    for (int i = n_new_dims - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    size_t offset = 0;
    for (auto operation : operations) {
        old_idx = std::get<0>(operation);
        new_idx = std::get<1>(operation);
        auto length = std::get<2>(operation);

        int64_t begin_old_idx = 0;
        int64_t begin_new_idx = 0;

        for (size_t i = 0; i < n_old_dims - 1; ++i) {
            begin_old_idx += old_idx[i] * old_strides[i];
        }
        begin_old_idx += old_idx.back();

        for (size_t i = 0; i < n_new_dims - 1; ++i) {
            begin_new_idx += new_idx[i] * new_strides[i];
        }
        begin_new_idx += new_idx.back();

        auto *old_indexer_data = (int64_t *) old_indexer.mutable_data() + offset;
        auto *new_indexer_data = (int64_t *) new_indexer.mutable_data() + offset;
        auto *mask_data = (uint8_t *) (mask.mutable_data() + begin_new_idx);
        for (int64_t i = 0; i < length; ++i) {
            *(mask_data + i) = 1;// TODO ? this only works because uint8_t is 1 byte
            *(old_indexer_data + i) = begin_old_idx + i;
            *(new_indexer_data + i) = begin_new_idx + i;
        }
        offset += length;
    }

    return {
            old_indexer,
            new_indexer,
            new_shape,
            mask.reshape(std::vector<int64_t>(new_shape.begin(), new_shape.begin() + n_new_dims))};
}


#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"

void flatten_py_list(
        py::list nested_list,
        std::vector<std::vector<int64_t>> &lengths,
        std::vector<std::tuple<std::vector<int64_t>, PyObject *>> &operations,
        std::vector<int64_t> &current_indices,
        std::vector<int64_t> &data_dim_map,
        std::vector<int64_t> &shape,
        int dim) {
    if (nested_list.empty()) {
        return;
    }

    bool is_data_dim = dim < data_dim_map.size() && data_dim_map[dim] >= 0;

    if (!py::isinstance<py::list>(nested_list[0])) {
        operations.emplace_back(current_indices, PySequence_Fast(nested_list.ptr(), "Something when wrong when reading one of the inner list"));
        current_indices.back() += nested_list.size();
    } else {

        if (lengths.size() <= dim + 1) {
            lengths.emplace_back();
        }

        for (auto &&i: nested_list) {

            auto sublist = i.cast<py::list>();
            lengths[dim + 1].push_back(sublist.size());

            if (is_data_dim) {
                current_indices.push_back(0);
            }

            flatten_py_list(
                    sublist,
                    lengths,
                    operations,
                    current_indices,
                    data_dim_map,
                    shape,
                    dim + 1);

            if (is_data_dim) {

                current_indices.pop_back();
                current_indices.back() += 1;

            } else {
                // current_indices.back() = offset;

                if (current_indices.size() > shape.size()) {
                    shape.push_back(0);
                }

                shape[current_indices.size() - 1] = std::max(shape[current_indices.size() - 1], current_indices.back());
            }
        }
    }

    if (is_data_dim) {
        shape[data_dim_map[dim]] = std::max(shape[data_dim_map[dim]], current_indices.back());
    }
}

#pragma clang diagnostic pop

std::tuple<py::array, py::array, std::vector<std::vector<int64_t>>> nested_py_list_to_padded_np_array(
        // nested list:
        const py::list &nested_list,
        // data dims:
        std::vector<int> data_dims,
        // dtype:
        py::dtype &dtype) {

    std::vector<std::vector<int64_t>> lengths;
    std::vector<std::tuple<std::vector<int64_t>, PyObject *>> operations;
    std::vector<int64_t> current_indices(1, 0);

    std::vector<int64_t> data_dim_map(*std::max_element(data_dims.begin(), data_dims.end()) + 1, -1);
    for (unsigned long i = 0; i < data_dims.size(); i++) {
        data_dim_map[data_dims[i]] = i;
    }

    lengths.emplace_back(1, nested_list.size());

    std::vector<int64_t> shape(data_dims.size(), 0);

    flatten_py_list(
            nested_list,
            lengths,
            operations,
            current_indices,
            data_dim_map,
            shape,
            0);

    // dtype_inference_break:

    py::array padded_array = py::array(py::dtype(dtype), shape);
    padded_array[py::make_tuple(py::ellipsis())] = 0;
    py::array padded_mask = py::array(py::dtype::of<uint8_t>(), shape);
    padded_mask[py::make_tuple(py::ellipsis())] = 0;

    const py::ssize_t *array_strides = padded_array.strides();
    const py::ssize_t *mask_strides = padded_mask.strides();

    for (const auto &op: operations) {
        auto indices = std::get<0>(op);
        auto flat_list = std::get<1>(op);

        char *array_data = (char *) padded_array.mutable_data();
        auto *mask_data = (uint8_t *) padded_mask.mutable_data();
        for (size_t dim = 0; dim < indices.size(); ++dim) {
            array_data += array_strides[dim] * indices[dim];
            mask_data += mask_strides[dim] * indices[dim];
        }

        PyObject **items = PySequence_Fast_ITEMS(flat_list);
        const int items_count = PySequence_Fast_GET_SIZE(flat_list);
        for (int i = 0; i < items_count; ++i) {
            PyArray_SETITEM(padded_array.ptr(), array_data, items[i]);
            array_data += array_strides[indices.size() - 1];
            *mask_data = 1;
            mask_data += mask_strides[indices.size() - 1];
        }
        Py_DECREF(flat_list);
    }

    return std::make_tuple(padded_array, padded_mask, lengths);
}

PYBIND11_MODULE(_C, m) {
    m.def("make_refolding_indexers", &make_refolding_indexers, "Refolds the tensor into a different shape");
    m.def("nested_py_list_to_padded_array", &nested_py_list_to_padded_np_array, "Converts a nested Python list to a padded array");
}

#pragma clang diagnostic pop
