#
# Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
#

from __future__ import print_function

try:
  from setuptools import setup
  from setuptools import Extension
except ImportError:
  from distutils.core import setup
  from distutils.extension import Extension
import sysconfig

from Cython.Build import cythonize

import numpy
import os
import subprocess
import sys

win32_build = os.name == 'nt'


from utils import generate_eigen_pyx

this_path  = os.path.dirname(os.path.realpath(__file__))
src_files = ['eigen/c_eigen.pxd', 'eigen/c_eigen_private.pxd', 'include/eigen_wrapper.hpp']
src_files.extend(['utils/angleaxis.in.pyx', 'utils/generate_eigen_pyx.py', 'utils/quaternion.in.pyx'])
src_files = [ '{}/{}'.format(this_path, f) for f in src_files ]

generate_eigen_pyx(this_path + "/eigen", this_path + "/utils")

def GenExtension(name):
  pyx_src = name.replace('.', '/')
  cpp_src = pyx_src + '.cpp'
  pyx_src = pyx_src + '.pyx'
  ext_src = pyx_src
  include_dirs = [os.path.join(os.getcwd(), "include"), "@EIGEN3_INCLUDE_DIR@", numpy.get_include()]
  compile_args = ['-std=c++11']
  print(sys.platform)
  if win32_build:
      compile_args = ['-DWIN32']
  elif sys.platform == 'darwin':
    from platform import machine
    osx_arch = machine()
    os.environ["ARCHFLAGS"] = "-arch " + osx_arch
    compile_args += ["-arch", osx_arch]

  return Extension(name, [ext_src], extra_compile_args = compile_args, include_dirs = include_dirs)

extensions = [
  GenExtension('eigen.eigen')
]

packages = ['eigen']
data = ['__init__.py', 'c_eigen.pxd', 'eigen.pxd']

cython_c_compiler_launcher="@CYTHON_C_COMPILER_LAUNCHER@"
if cython_c_compiler_launcher:
  sysconfig.get_config_vars()["CC"] = cython_c_compiler_launcher + " " + sysconfig.get_config_vars()["BINDIR"] + "/" + sysconfig.get_config_vars()["CC"]
cython_cxx_compiler_launcher="@CYTHON_CXX_COMPILER_LAUNCHER@"
if cython_cxx_compiler_launcher:
  sysconfig.get_config_vars()["CXX"] = cython_cxx_compiler_launcher + " " + sysconfig.get_config_vars()["BINDIR"] + "/" + sysconfig.get_config_vars()["CXX"]

extensions = cythonize(extensions, cache = True)

dependencies = [
  "Cython>=0.2",
  "coverage",
  "numpy>=1.8.2",
  "pytest"
]

setup(
    name = 'eigen',
    version='@PROJECT_VERSION@',
    ext_modules = extensions,
    package_data = { 'eigen': data },
    install_requires=dependencies
)
