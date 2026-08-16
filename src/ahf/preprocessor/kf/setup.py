from setuptools import setup, Extension  # Changed from distutils.core
from Cython.Build import cythonize
import numpy

ext = Extension(
    "kalman_moving_average",
    sources=["kalman_moving_average.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize([ext], compiler_directives={'language_level': "3"})
)
