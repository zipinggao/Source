from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='compute_overlap',
    ext_modules=cythonize([
        Extension("compute_overlap", ["compute_overlap.pyx"]),
    ]),
)