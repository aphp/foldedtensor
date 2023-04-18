cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(foldedtensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYTHON_VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv" CACHE PATH "Path to Python virtual environment")

# Specify the Python virtual environment path
set(Python_ROOT_DIR "${PYTHON_VENV_PATH}")

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Execute a Python script to find torch include and library directories
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(';'.join(cpp_extension.include_paths()), end='')"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0], end='')"
        OUTPUT_VARIABLE TORCH_LIBRARY_DIR
)

include_directories(${Python_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARY_DIR})

add_library(foldedtensor_functions SHARED
        foldedtensor/functions.cpp
        )

set(TORCH_LIBRARIES torch torch_cpu c10)
if (WIN32)
    list(APPEND TORCH_LIBRARIES torch_python)
endif()

target_link_libraries(foldedtensor_functions PRIVATE "${TORCH_LIBRARIES}" "${PYTHON_LIBRARIES}")

set_target_properties(foldedtensor_functions PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/lib"
        )

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET foldedtensor_functions
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:foldedtensor_functions>
            )
endif()
