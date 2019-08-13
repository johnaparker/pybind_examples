cmake_minimum_required(VERSION 3.1)
project(foo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-Werror -Og")

add_subdirectory(pybind11)

set(SOURCES ${SOURCES} 
    src/special.cpp
)

set(BINDINGS
    src/main_pyb.cpp
    src/special_pyb.cpp
)

add_library(foo MODULE "${SOURCES}" "${BINDINGS}")
target_link_libraries(foo PRIVATE pybind11::module)
set_target_properties(foo PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")
