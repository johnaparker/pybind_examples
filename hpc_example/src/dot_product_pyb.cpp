#include "dot_product.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

void bind_dot_product(py::module &m) {
    m.def("dot_product", dot_product, 
           "x"_a, "y"_a, R"pbdoc(
        dot product of x and y (serial)
    )pbdoc");
}

void bind_omp_dot_product(py::module &m) {
    m.def("omp_dot_product", omp_dot_product, 
           "x"_a, "y"_a, R"pbdoc(
        dot product of x and y (OpenMP)
    )pbdoc");
}

void bind_mpi_dot_product(py::module &m) {
    m.def("mpi_dot_product", mpi_dot_product, 
           "x"_a, "y"_a, R"pbdoc(
        dot product of x and y (MPI)
    )pbdoc");
}
