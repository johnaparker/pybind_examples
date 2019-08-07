#ifndef GUARD_dot_product_h
#define GUARD_dot_product_h

#include <eigen3/Eigen/Core>

using Array = Eigen::Array<double, Eigen::Dynamic, 1>;
using Eigen::Ref;

double dot_product(const Ref<const Array> &x, const Ref<const Array> &y);
double omp_dot_product(const Ref<const Array> &x, const Ref<const Array> &y);
double mpi_dot_product(const Ref<const Array> &x, const Ref<const Array> &y);

double gpu_full_dot_product(const double *a, const double *b, int N);
double gpu_dot_product(const Ref<const Array> &x, const Ref<const Array> &y);

#endif
