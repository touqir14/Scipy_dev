from distutils.core import setup
from Cython.Build import cythonize

setup(
  ext_modules=cythonize(["rref_cy.pyx"], annotate=True),
  include_dirs=['C:/ProgramData/Miniconda3/pkgs/numpy-1.14.3-py36_intel_0/Lib/site-packages/numpy/core/include'],
)