import distutils.command.clean
import glob
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


version = "0.1.0"
sha = "Unknown"
package_name = "foldedtensor"

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    pass

print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, "foldedtensor", "version.py")
    with open(version_path, "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
        f.write('git_version = "{}"\n'.format(sha))


write_version_file()

readme = open("README.md").read()

pytorch_dep = "torch"

if os.getenv("PYTORCH_VERSION"):
    pytorch_dep += "==" + os.getenv("PYTORCH_VERSION")
else:
    pytorch_dep += ">=1.7.0"

requirements = [pytorch_dep]


def get_extensions():
    import torch
    from torch.utils.cpp_extension import (
        CUDA_HOME,
        CppExtension,
        CUDAExtension,
    )

    extension = CppExtension

    define_macros = []

    extra_link_args = []
    extra_compile_args = {"cxx": ["-O3", "-g", "-std=c++14"]}
    if int(os.environ.get("DEBUG", 0)):
        extra_compile_args = {"cxx": ["-O0", "-fno-inline", "-g", "-std=c++14"]}
        extra_link_args = ["-O0", "-g"]
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args["nvcc"] = nvcc_flags

    if sys.platform == "win32":
        define_macros += [("foldedtensor_EXPORTS", None)]

    src_dir = Path(__file__).parent / "foldedtensor"

    sources = set(str(p.absolute()) for p in src_dir.glob("*.cpp"))

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        sources |= set(str(p.absolute()) for p in src_dir.glob("*.cu"))

    ext_modules = [
        extension(
            "foldedtensor._C",
            list(sources),
            include_dirs=[src_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


class lazy_build_ext(build_ext):
    def run(self):
        from torch.utils.cpp_extension import (
            BuildExtension,
        )

        self.distribution.ext_modules = get_extensions()

        build_ext_instance = BuildExtension(
            self.distribution,
            no_python_abi_suffix=True,
            use_ninja=os.environ.get("USE_NINJA", False),
        )
        # For editable installs, we need to pass inplace
        build_ext_instance.inplace = self.inplace

        build_ext_instance.finalize_options()

        return build_ext_instance.run()


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore", "r") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


setuptools.setup(
    name=package_name,
    version=version,
    author="Perceval Wajsbürt",
    author_email="perceval.wajsburt-ext@aphp.fr",
    description="FoldedTensors for PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/percevalw/foldedtensor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
    cmdclass={
        "clean": clean,
        "build_ext": lazy_build_ext,
    },
    install_requires=requirements,
    ext_modules=[setuptools.Extension("foldedtensor._C", [])],
    package_data={"": ["*.cpp", ".so", ".dylib", ".dll"]},
)
