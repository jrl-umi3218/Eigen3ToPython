#
# Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
#

cmake_minimum_required(VERSION 2.8)

find_package(Eigen3 REQUIRED)
message("Found Eigen3: ${EIGEN3_VERSION_STRING}")
message("Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
configure_file(setup.in.py ${CMAKE_CURRENT_SOURCE_DIR}/setup.py)

option(PYTHON_BINDING_USER_INSTALL "Install the Python bindings in user space" OFF)
option(DISABLE_TESTS "Disable unit tests." OFF)

set(PIP_EXTRA_OPTIONS "")
if(${PYTHON_BINDING_USER_INSTALL})
  set(PIP_EXTRA_OPTIONS "--user")
endif()
if(DEFINED PIP_INSTALL_PREFIX)
  set(PIP_EXTRA_OPTIONS --prefix ${PIP_INSTALL_PREFIX})
endif()

# Install requirements
add_custom_target(eigen-python-bindings-requirements ALL
  COMMAND pip install ${PIP_EXTRA_OPTIONS} -r requirements.txt
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
          setup.in.py
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
