#include "special.hpp"
#include <cmath>

double cpp_factorial(double n) {
    return std::tgamma(n+1);
}
