cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-Werror -Og")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -lineinfo")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
add_subdirectory(pybind11)

set(SOURCES ${SOURCES} 
    src/dot_product.cpp
)

set(CUDA_SOURCES
    src/dot_product_gpu.cu
)

set(BINDINGS
    src/main_pyb.cpp
    src/dot_product_pyb.cpp
)

add_library(gpu STATIC "${CUDA_SOURCES}")
set_target_properties(gpu PROPERTIES CUDA_SEPERABLE_COMPILATION ON
                                     POSITION_INDEPENDENT_CODE TRUE) 

add_library(cpp MODULE "${SOURCES}" "${BINDINGS}")
target_link_libraries(cpp PRIVATE pybind11::module OpenMP::OpenMP_CXX Eigen3::Eigen ${MPI_LIBRARIES} gpu)
set_target_properties(cpp PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")
