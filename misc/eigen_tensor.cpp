/*
Compile with:

    g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` eigen_tensor.cpp -o mod`python3-config --extension-suffix` -I/path/to/pybind11/include -I/usr/include/python3.7m

*/

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <vector>

using Tensor2 = Eigen::Tensor<double,2,Eigen::RowMajor>;
using Tensor3 = Eigen::Tensor<double,3,Eigen::RowMajor>;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Eigen::Ref;
namespace py = pybind11;

double sum_1(const Ref<const Matrix>& M) {
    double sum = 0;
    for (int i = 0; i < M.rows(); i++) {
        for (int j = 0; j < M.rows(); j++) {
            sum += M(i,j);
        }
    }

    return sum;
}

// this one is slower than the others due to data-copying M
double sum_2(const Matrix& M) {
    double sum = 0;
    for (int i = 0; i < M.rows(); i++) {
        for (int j = 0; j < M.rows(); j++) {
            sum += M(i,j);
        }
    }

    return sum;
}

double sum_3(py::array_t<double> M) {
    double sum = 0;

    auto r = M.mutable_unchecked<2>();
    double *data = r.mutable_data(0,0);
    Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::RowMajor>> M_tensor(data, r.shape(0), r.shape(1));

    for (int i = 0; i < M_tensor.dimension(0); i++) {
        for (int j = 0; j < M_tensor.dimension(1); j++) {
            sum += M_tensor(i,j);
        }
    }

    return sum;
}

double sum_4(const py::array_t<double> M) {
    double sum = 0;

    const auto r = M.unchecked<2>();
    const Eigen::TensorMap<Eigen::Tensor<const double, 2, Eigen::RowMajor>> M_tensor(r.data(0,0), r.shape(0), r.shape(1));

    for (int i = 0; i < M_tensor.dimension(0); i++) {
        for (int j = 0; j < M_tensor.dimension(1); j++) {
            sum += M_tensor(i,j);
        }
    }

    return sum;
}

double sum_5(const py::array_t<double> M) {
    double sum = 0;
    auto r = M.unchecked<2>();

    for (int i = 0; i < r.shape(0); i++) {
        for (int j = 0; j < r.shape(1); j++) {
            sum += r(i,j);
        }
    }

    return sum;
}

PYBIND11_MODULE(mod, m) {
    m.doc() = "pybind11 example plugin";
    m.def("sum_1", &sum_1, "sum over");
    m.def("sum_2", &sum_2, "sum over");
    m.def("sum_3", &sum_3, "sum over");
    m.def("sum_4", &sum_4, "sum over");
    m.def("sum_5", &sum_5, "sum over");
}
