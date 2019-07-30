#define NOMINMAX
#include <pybind11/pybind11.h>

namespace py = pybind11;

// special submodule
void bind_cpp_factorial(py::module &);

PYBIND11_MODULE(cpp, m) {
    m.doc() = R"pbdoc(
        C++ submodule of foo
        -----------------------

        .. currentmodule:: cpp

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // special submodule
    py::module special_m = m.def_submodule("special", "special functions module");
    bind_cpp_factorial(special_m);
}
