cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(foldedtensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the python virtual environment and set the Python_ROOT_DIR to it
set(PYTHON_VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv" CACHE PATH "Path to Python virtual environment")
set(Python_ROOT_DIR "${PYTHON_VENV_PATH}")

# Find Python with numpy
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

# Find pybind11
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir(), end='')"
        OUTPUT_VARIABLE pybind11_DIR
)
find_package(pybind11 CONFIG REQUIRED)

add_library(
        foldedtensor_functions SHARED
        foldedtensor/functions.cpp
)

target_link_libraries(foldedtensor_functions Python::NumPy pybind11::headers "${PYTHON_LIBRARIES}")

set_target_properties(foldedtensor_functions PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/lib"
        )
