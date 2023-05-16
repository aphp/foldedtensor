import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "foldedtensor._C",
        ["foldedtensor/functions.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            np.get_include(),
        ],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
