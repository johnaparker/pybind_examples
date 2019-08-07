#include "dot_product.hpp"
#include <chrono>
#include <functional>
#include <mpi.h>
#include <iostream>
#include <stdio.h>

using Array = Eigen::Array<double, Eigen::Dynamic, 1>;
using Eigen::Ref;

double dot_product(const Ref<const Array> &x, const Ref<const Array> &y) {
    const int size = x.size();
    double sum = 0;

    const double *x_p = x.data();
    const double *y_p = y.data();

    #pragma omp simd reduction(+: sum)
    for (int i = 0; i < size; i++) {
        sum += x_p[i]*y_p[i];
    }

    return sum;
    //return x.matrix().dot(y.matrix());
}

double omp_dot_product(const Ref<const Array> &x, const Ref<const Array> &y) {
    int size = x.size();
    double sum = 0;

    const double *x_p = x.data();
    const double *y_p = y.data();

    #pragma omp parallel for simd shared(x, y) reduction(+: sum)
    for (int i = 0; i < size; i++) {
        sum += x_p[i]*y_p[i];
    }

    return sum;
}

double mpi_dot_product(const Ref<const Array> &x, const Ref<const Array> &y) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int L = floor(x.size()/double(size));
    int ida = rank*L;
    int idb;

    if (rank != size - 1)
        idb = ida + L;
    else
        idb = x.size();

    double sum = 0;
    double final_sum;

    const double *x_p = x.data();
    const double *y_p = y.data();

    #pragma omp simd reduction(+: sum)
    for (int i = ida; i < idb; i++) {
        sum += x_p[i]*y_p[i];
    }

    MPI_Reduce(&sum, &final_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&final_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return final_sum;
}

double gpu_dot_product(const Ref<const Array> &x, const Ref<const Array> &y) {
    int size = x.size();
    const double *x_p = x.data();
    const double *y_p = y.data();

    double sum = gpu_full_dot_product(x_p, y_p, size);
    return sum;
}
