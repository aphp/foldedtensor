#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"

#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/python_scalars.h>
#include <torch/script.h>
#include <vector>

#include <pybind11/pytypes.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
// #include <torch/csrc/utils/python_symnode.h>

// #include <ATen/PythonTorchFunctionTLS.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// #include <c10/core/SymNodeImpl.h>

namespace py = pybind11;
using namespace torch::indexing;


std::tuple<torch::Tensor, at::optional<torch::Tensor>> refold(
        const torch::Tensor &data,
        std::vector<std::vector<int>> lengths,
        std::vector<int> old_data_dims,
        std::vector<int> new_data_dims,
        bool return_mask) {
    unsigned int n_lengths = lengths.size();
    unsigned int n_old_dims = old_data_dims.size();
    unsigned int n_new_dims = new_data_dims.size();

    std::vector<int> old_dim_map(n_lengths, -1);
    std::vector<int> new_dim_map(n_lengths, -1);

    std::vector<int64_t> old_idx(n_old_dims, 0);
    std::vector<int64_t> new_idx(n_new_dims, 0);

    std::vector<std::tuple<
            std::vector<int64_t>,
            std::vector<int64_t>,
            int64_t>>
            operations;
    operations.reserve(lengths.back().size());

    std::vector<int64_t> shape(n_new_dims + data.dim() - n_old_dims, 0);

    // Copy last dimensions from data shape to new `shape`
    for (unsigned long i = 0; i < data.dim() - n_old_dims; i++) {
        shape[n_new_dims + i] = data.size(n_old_dims + i);
    }

    for (unsigned long i = 0; i < n_old_dims; i++) {
        old_dim_map[old_data_dims[i]] = i;
    }

    for (unsigned long i = 0; i < n_new_dims; i++) {
        new_dim_map[new_data_dims[i]] = i;
    }

    std::vector<int> offsets(n_lengths - 1, 0);
    for (int length: lengths.back()) {
        operations.emplace_back(old_idx, new_idx, length);

        new_idx.back() += length;
        old_idx.back() += length;
        shape[n_new_dims - 1] = std::max(shape[n_new_dims - 1], new_idx.back());

        int dim = old_dim_map.size() - 2;
        if (old_dim_map[dim] >= 0) {
            old_idx[old_dim_map[dim]] += 1;
            for (unsigned long i = old_dim_map[dim] + 1; i < n_old_dims; i++) {
                old_idx[i] = 0;
            }
        }

        if (new_dim_map[dim] >= 0) {
            new_idx[new_dim_map[dim]] += 1;
            shape[new_dim_map[dim]] = std::max(shape[new_dim_map[dim]], new_idx[new_dim_map[dim]]);
            for (unsigned long i = new_dim_map[dim] + 1; i < n_new_dims; i++) {
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
            int next_old_data_dim = old_dim_map[next_dim];
            if (next_old_data_dim >= 0) {
                old_idx[next_old_data_dim] += 1;
                for (unsigned long i = next_old_data_dim + 1; i < n_old_dims; i++) {
                    old_idx[i] = 0;
                }
            }

            int next_new_data_dim = new_dim_map[next_dim];

            if (next_new_data_dim >= 0) {
                new_idx[next_new_data_dim] += 1;
                shape[next_new_data_dim] = std::max(shape[next_new_data_dim], new_idx[next_new_data_dim]);

                // DEBUG ASSERT NEVER 3

                for (unsigned long i = next_new_data_dim + 1; i < n_new_dims; i++) {
                    new_idx[i] = 0;
                }
            }
        }
    }

    std::vector<int64_t> data_flat_dims = {-1};
    for (int64_t i = n_old_dims; i < data.dim(); ++i) {
        data_flat_dims.push_back(data.size(i));
    }
    torch::Tensor old_data_flat = data.view(data_flat_dims);

    std::vector<int64_t> new_data_flat_dims = {-1};
    for (int64_t i = n_new_dims; i < shape.size(); ++i) {
        new_data_flat_dims.push_back(shape[i]);
    }
    torch::Tensor new_data = torch::zeros(shape, data.options());
    torch::Tensor new_data_flat = new_data.view(new_data_flat_dims);

    at::optional<torch::Tensor> mask_opt;

    if (return_mask) {
        mask_opt = torch::zeros(
                {new_data_flat.size(0)},
                torch::TensorOptions().dtype(torch::kByte));
    }

    // Init new strides (full of 1, size = n_old_dims) and compute them in reverse for data size and new data sizes
    // from n_new_dims and n_old_dims offsets
    std::vector<long long> old_strides(n_old_dims, 1);
    for (int i = n_old_dims - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * data.size(i + 1);
    }
    std::vector<long long> new_strides(n_new_dims, 1);
    for (int i = n_new_dims - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * shape[i + 1];
    }

    for (const auto &operation: operations) {
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

        // print(begin_old_idx, begin_new_idx, length)

        if (return_mask) {
            auto *mask_data = (uint8_t *) mask_opt.value().data_ptr();

            for ([[maybe_unused]] const auto i: at::irange(length)) {
                *mask_data = 1;
                mask_data += 1;
            }
        }

        auto old_slice = old_data_flat.narrow(0, begin_old_idx, length);
        new_data_flat.narrow(0, begin_new_idx, length).copy_(old_slice);
    }

    new_data = new_data.view(shape);
    if (return_mask) {
        mask_opt.value() = (mask_opt.value()
                                    .to(new_data.device())
                                    .view(std::vector<int64_t>(shape.begin(), shape.begin() + n_new_dims)));
    }

    return {new_data, mask_opt};
}


at::ScalarType infer_scalar_type(PyObject *obj) {
    /*if (torch::is_symint(obj)) {
        return at::ScalarType::Long;
    }
    if (torch::is_symfloat(obj)) {
        return at::ScalarType::Double;
    }*/
#ifdef USE_NUMPY
    if (is_numpy_available()) {
        if (PyArray_Check(obj)) {
            return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject *) obj));
        }
        if (PyArray_CheckScalar(obj)) {
            THPObjectPtr arr(PyArray_FromScalar(obj, nullptr));
            return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject *) arr.get()));
        }
    }
#endif
    if (PyFloat_Check(obj)) {
        // this is always guaranteed to be a floating-point type, and makes it more
        // convenient to write e.g. torch.tensor(0.) than torch.tensor(0.,
        // dtype=torch.Tensor.dtype).
        return torch::tensors::get_default_scalar_type();
    }
    if (THPUtils_checkLong(obj)) {
        return at::ScalarType::Long;
    }
    if (PyBool_Check(obj)) {
        return at::ScalarType::Bool;
    }
    if (PyComplex_Check(obj)) {
        switch (torch::tensors::get_default_scalar_type()) {
            case at::ScalarType::Float:
                return at::ScalarType::ComplexFloat;
            case at::ScalarType::Double:
                return at::ScalarType::ComplexDouble;
            default:
                TORCH_CHECK(false, "invalid default scalar type for complex")
        }
    }
    if (THPVariable_Check(obj)) {
        const auto &var = THPVariable_Unpack(obj);
        return var.scalar_type();
    }
    if (THPUtils_checkString(obj)) {
        throw torch::TypeError("new(): invalid data type '%s'", Py_TYPE(obj)->tp_name);
    }
    AT_ERROR("Could not infer dtype of ", Py_TYPE(obj)->tp_name);
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

std::tuple<torch::Tensor, torch::Tensor, std::vector<std::vector<int64_t>>> nested_py_list_to_padded_tensor(
        const py::list &nested_list,
        std::vector<int> data_dims) {
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

    at::optional<at::ScalarType> scalar_type;
    for (const auto &op: operations) {
        auto items = std::get<1>(op);
        for (const auto i: at::irange(PySequence_Fast_GET_SIZE(items))) {
            at::ScalarType item_scalar_type = infer_scalar_type(PySequence_Fast_GET_ITEM(items, i));
            scalar_type = (scalar_type ? at::promoteTypes(*scalar_type, item_scalar_type)
                                       : item_scalar_type);

            // Completely break the scalar_type inference loop using goto
            if (scalar_type == at::ScalarType::ComplexDouble) {
                goto scalar_type_inference_break;
            }
        }
    }

scalar_type_inference_break:

    torch::Tensor padded_tensor = torch::zeros(shape, torch::TensorOptions().dtype(
                                                              scalar_type ? *scalar_type : torch::tensors::get_default_scalar_type()));
    torch::Tensor padded_mask = torch::zeros(shape, torch::TensorOptions().dtype(torch::kByte));

    at::IntArrayRef strides = padded_tensor.strides();
    at::ScalarType dtype = padded_tensor.scalar_type();
    size_t element_tensor_size = padded_tensor.element_size();

    for (const auto &op: operations) {
        auto indices = std::get<0>(op);
        auto flat_list = std::get<1>(op);

        char *tensor_data = (char *) padded_tensor.data_ptr();
        auto *mask_data = (uint8_t *) padded_mask.data_ptr();
        for (size_t dim = 0; dim < indices.size(); ++dim) {
            tensor_data += strides[dim] * element_tensor_size * indices[dim];
            mask_data += strides[dim] * indices[dim];
        }

        PyObject **items = PySequence_Fast_ITEMS(flat_list);
        for (const auto i: at::irange(PySequence_Fast_GET_SIZE(flat_list))) {
            torch::utils::store_scalar(tensor_data, dtype, items[i]);
            tensor_data += strides[indices.size() - 1] * element_tensor_size;
            *mask_data = 1;
            mask_data += strides[indices.size() - 1];
        }
        Py_DECREF(flat_list);
    }

    return std::make_tuple(padded_tensor, padded_mask, lengths);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("refold", &refold, "Refolds the tensor into a different shape");
    m.def("nested_py_list_to_padded_tensor", &nested_py_list_to_padded_tensor, "Converts a nested Python list to a padded tensor");
}

#pragma clang diagnostic pop
