
import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++17', '-stdlib=libc++']

sfc_module = Extension(
    'superfastcode2', sources = [''],
    include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args = cpp_args,
)

setup(
    name = 'madml',
    version = '0.001a',
    description = 'Python package for madml',
    ext_modules = [sfc_module],
)