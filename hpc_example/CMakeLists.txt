cmake_minimum_required(VERSION 3.1)
project(cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("Building in Release")
endif()
message("${CMAKE_BUILD_TYPE}")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -funroll-loops")

find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_subdirectory(pybind11)
#include_directories(src)

set(SOURCES ${SOURCES} 
    src/dot_product.cpp
)

set(BINDINGS
    src/main_pyb.cpp
    src/dot_product_pyb.cpp
)

add_library(cpp MODULE "${SOURCES}" "${BINDINGS}")
target_link_libraries(cpp PRIVATE pybind11::module)
target_link_libraries(cpp PRIVATE pybind11::module OpenMP::OpenMP_CXX Eigen3::Eigen)
set_target_properties(cpp PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")
