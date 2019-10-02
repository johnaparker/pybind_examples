/*
Compile with:

g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` interp.cpp -o interp`python3-config --extension-suffix` -I/path/to/pybind11/include -I/usr/include/python3.7m

Run speed test:

python interp_perf.py

*/

#include "math.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <array>
#include <complex>

namespace py = pybind11;

using dtype = std::complex<double>;
using arr_in = py::array_t<double>;
using arr_out = py::array_t<dtype>;
using namespace py::literals;

arr_out grid_interpolate(const std::array<arr_in,2> &grid, const arr_out &data, const arr_in &pts, dtype fill_value = 0) {
    auto x_p = grid[0].unchecked<1>();
    auto y_p = grid[1].unchecked<1>();

    double x0_grid = x_p(0);
    double y0_grid = y_p(0);
    double dx_grid = x_p(1) - x_p(0);
    double dy_grid = y_p(1) - y_p(0);

    auto data_p = data.unchecked<3>();
    auto pts_p = pts.unchecked<2>();

    int Npts = pts_p.shape(0);
    int ndim_out = data.shape(2);
    auto result = arr_out({Npts, ndim_out});
    auto result_p = result.mutable_unchecked<2>();

    for (int i = 0; i < Npts; i++) {
        int x1 = std::floor((pts_p(i,0) - x0_grid) / dx_grid);
        int y1 = std::floor((pts_p(i,1) - y0_grid) / dy_grid);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        if (x1 < 0 || y1 < 0 || x2 > x_p.size() || y2 > y_p.size()) {
            for (int j = 0; j < ndim_out; j++) {
                result_p(i,j) = fill_value;
            }
        }
        else {
            double dx = pts_p(i,0) - x_p(x1);
            double dy = pts_p(i,1) - y_p(y1);

            for (int j = 0; j < ndim_out; j++) {
                result_p(i,j) = data_p(x1, y1, j)*(1 - dx/dx_grid)*(1 - dy/dy_grid)
                              + data_p(x2, y2, j)*(dx/dx_grid)*(dy/dy_grid)
                              + data_p(x1, y2, j)*(1 - dx/dx_grid)*(dy/dy_grid)
                              + data_p(x2, y1, j)*(dx/dx_grid)*(1 - dy/dy_grid);
            }
        }
    }

    return result;
}

PYBIND11_MODULE(interp, m) {
    m.doc() = "pybind11 example plugin";
    m.def("grid_interpolate", grid_interpolate, 
            "grid"_a, "data"_a, "pts"_a, "fill_value"_a=0, R"pbdoc(
        interpolate data on a grid
    )pbdoc");
}
