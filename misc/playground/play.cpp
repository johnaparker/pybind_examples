#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

void f() {
    return;
}

PYBIND11_MODULE(play, m) {
    m.def("f", f);
}
