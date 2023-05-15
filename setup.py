import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# The C++ extension module
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
    name="foldedtensor",
    version="0.1",
    author="Your Name",
    author_email="you@example.com",
    description="A test project using pybind11 and NumPy",
    long_description="",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.5.0"],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
