# Copyright 2012-2017 CNRS-UM LIRMM, CNRS-AIST JRL
#
# This file is part of Eigen3ToPython.
#
# Eigen3ToPython is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Eigen3ToPython is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Eigen3ToPython.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
try:
  from setuptools import setup
  from setuptools import Extension
except ImportError:
  from distutils.core import setup
  from distutils.extension import Extension

from Cython.Build import cythonize
import os
import subprocess

win32_build = os.name == 'nt'

from utils import generate_eigen_pyx

this_path  = os.path.dirname(os.path.realpath(__file__))
version_hash = generate_eigen_pyx(this_path + "/eigen", this_path + "/utils")

class pkg_config(object):
  def __init__(self, package):
    self.compile_args = []
    self.include_dirs = []
    self.library_dirs = []
    self.libraries = []
    self.found = True
    self.name = package
    try:
      tokens = subprocess.check_output(['pkg-config', '--libs', '--cflags', package]).split()
      tokens = [ token.decode('ascii') for token in tokens ]
    except subprocess.CalledProcessError:
      tokens = []
      self.found = False
    for token in tokens:
      flag = token[:2]
      value = token[2:]
      if flag == '-I':
        self.include_dirs.append(value)
      elif flag == '-l':
        if value[0] == ':':
          value = value[1:]
        self.libraries.append(value)
      elif flag == '-L':
        self.library_dirs.append(value)
      else:
        if win32_build:
          if token[len(token)-4:] == '.lib':
            self.libraries.append(token[:len(token)-4])
          elif token[:9] == '/LIBPATH:':
            self.library_dirs.append(token[9:])
          else:
            self.compile_args.append(token)
        else:
          self.compile_args.append(token)
  def __repr__(self):
    return str(self.include_dirs)+", "+str(self.library_dirs)+", "+str(self.libraries)

python_libs = []
python_lib_dirs = []
python_others = []
if not win32_build:
  tokens = subprocess.check_output(['python-config', '--ldflags']).split()
  tokens = [ token.decode('ascii') for token in tokens ]
  for token in tokens:
    flag = token[:2]
    value = token[2:]
    if flag == '-l':
      python_libs.append(value)
    elif flag == '-L':
      python_lib_dirs.append(value)
    elif token[:1] == '-':
      python_others.append(token)

configs = { pkg: pkg_config(pkg) for pkg in ['eigen3'] }

for p,c in configs.items():
  c.compile_args.append('-std=c++11')
  for o in python_others:
    c.compile_args.append(o)
  c.include_dirs.append(os.getcwd() + "/include")
  if not win32_build:
    c.library_dirs.extend(python_lib_dirs)
    c.libraries.extend(python_libs)
  else:
    c.compile_args.append("-DWIN32")
  if p != 'eigen3':
    c.include_dirs.extend(configs['eigen3'].include_dirs)

def GenExtension(name, pkg, ):
  pyx_src = name.replace('.', '/')
  cpp_src = pyx_src + '.cpp'
  pyx_src = pyx_src + '.pyx'
  ext_src = pyx_src
  if pkg.found:
    return Extension(name, [ext_src], extra_compile_args = pkg.compile_args, include_dirs = pkg.include_dirs, library_dirs = pkg.library_dirs, libraries = pkg.libraries)
  else:
    print("Failed to find {}".format(pkg.name))
    return None

extensions = [
  GenExtension('eigen.eigen', configs['eigen3'])
]

extensions = [ x for x in extensions if x is not None ]
packages = ['eigen']
data = ['__init__.py', 'c_eigen.pxd', 'eigen.pxd', 'eigen.so']

cython_packages = [ x for x in packages if any([ext.name.startswith(x) for ext in extensions]) ]

extensions = cythonize(extensions)

setup(
    name = 'eigen',
    version='1.0.0-{}'.format(version_hash),
    ext_modules = extensions,
    packages = packages,
    package_data = { 'eigen': data }
)
