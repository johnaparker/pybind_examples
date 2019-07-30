#define NOMINMAX
#include <pybind11/pybind11.h>

namespace py = pybind11;

// special submodule
void bind_cpp_factorial(py::module &);

PYBIND11_MODULE(foo, m) {
    m.doc() = R"pbdoc(
        C++ foo module
        -----------------------

        .. currentmodule:: foo

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // special submodule
    py::module special_m = m.def_submodule("special", "special functions module");
    bind_cpp_factorial(special_m);
}
