cmake_minimum_required(VERSION 3.1)
project(cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("Building in Release")
endif()
message("${CMAKE_BUILD_TYPE}")


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${MPI_INCLUDE_PATH})
add_subdirectory(pybind11)

set(SOURCES ${SOURCES} 
    src/dot_product.cpp
)

set(BINDINGS
    src/main_pyb.cpp
    src/dot_product_pyb.cpp
)

#target_include_directories(my_mpi_bin PRIVATE ${MPI_CXX_INCLUDE_PATH} src1.h src2.h)
#target_compile_options(my_mpi_bin PRIVATE ${MPI_CXX_COMPILE_FLAGS} my_compile_flags)
#target_link_libraries(my_mpi_bin ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} my_link_flags)

add_library(cpp MODULE "${SOURCES}" "${BINDINGS}")
target_link_libraries(cpp PRIVATE pybind11::module OpenMP::OpenMP_CXX Eigen3::Eigen ${MPI_LIBRARIES})
set_target_properties(cpp PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")
