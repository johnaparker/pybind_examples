#include "dot_product.hpp"
#include <iostream>
#include <chrono>
#include <functional>

using Array = Eigen::Array<double, Eigen::Dynamic, 1>;
using Eigen::Ref;

double dot_product(const Ref<const Array> &x, const Ref<const Array> &y) {
    const int size = x.size();
    double ret = 0;

    const double *x_p = x.data();
    const double *y_p = y.data();

#pragma omp simd reduction(+: ret)
    for (int i = 0; i < size; i++) {
        ret += x_p[i]*y_p[i];
    }

    return ret;
    //return x.matrix().dot(y.matrix());
}

double omp_dot_product(const Ref<const Array> &x, const Ref<const Array> &y) {
    int size = x.size();
    double ret = 0;

    const double *x_p = x.data();
    const double *y_p = y.data();

    #pragma omp parallel for simd shared(x, y) reduction( + : ret )
    for (int i = 0; i < size; i++) {
        ret += x_p[i]*y_p[i];
    }

    return ret;
}

double mpi_dot_product(const Ref<const Array> &x, const Ref<const Array> &y) {
    int size = x.size();
    double ret = 0;

    const double *a = x.data();
    const double *b = y.data();

#pragma omp simd reduction(+: ret)
    for (int i = 0; i < size; i++) {
        ret += a[i]*b[i];
    }

    return ret;
}
