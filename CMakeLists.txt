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

cmake_minimum_required(VERSION 2.8)

option(PYTHON_BINDING_USER_INSTALL "Install the Python bindings in user space" OFF)
option(DISABLE_TESTS "Disable unit tests." OFF)

set(PIP_EXTRA_OPTIONS "")
if(${PYTHON_BINDING_USER_INSTALL})
  set(PIP_EXTRA_OPTIONS "--user")
endif()

# Install requirements
add_custom_target(eigen-python-bindings-requirements ALL
  COMMAND pip install -r requirements.txt ${PIP_EXTRA_OPTIONS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS requirements.txt
)

# Build the bindings locally
add_custom_target(eigen-python-bindings ALL
  COMMAND python setup.py build_ext --inplace
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating Eigen Python bindings"
  DEPENDS eigen-python-bindings-requirements
          include/eigen_wrapper.hpp
          eigen/c_eigen.pxd
          eigen/c_eigen_private.pxd
          setup.py
          utils/angleaxis.in.pyx
          utils/generate_eigen_pyx.py
          utils/quaternion.in.pyx
)

# Test the bindings
if(NOT ${DISABLE_TESTS})
  enable_testing()
  add_test(NAME EigenPythonBindingsTest
    COMMAND nosetests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Install the bindings
add_custom_target(install-eigen-python-bindings
  COMMAND pip install . ${PIP_EXTRA_OPTIONS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Install Eigen Python bindings"
)
install(CODE "execute_process(COMMAND ${CMAKE_MAKE_PROGRAM} install-eigen-python-bindings)")
