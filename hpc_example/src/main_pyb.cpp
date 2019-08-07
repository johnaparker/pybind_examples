#define NOMINMAX
#include <pybind11/pybind11.h>

namespace py = pybind11;

// special submodule
void bind_dot_product(py::module &);
void bind_omp_dot_product(py::module &);
void bind_mpi_dot_product(py::module &);
void bind_gpu_dot_product(py::module &);

PYBIND11_MODULE(cpp, m) {
    m.doc() = R"pbdoc(
        C++ submodule of dot_product
        -----------------------

        .. currentmodule:: cpp

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    bind_dot_product(m);
    bind_omp_dot_product(m);
    bind_mpi_dot_product(m);
    bind_gpu_dot_product(m);
}
