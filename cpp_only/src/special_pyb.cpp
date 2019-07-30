#include "special.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

void bind_cpp_factorial(py::module &m) {
    m.def("cpp_factorial", py::vectorize(cpp_factorial), 
           "n"_a, R"pbdoc(
        Compute the factorial of n
    )pbdoc");
}
