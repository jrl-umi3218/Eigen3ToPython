#
# Copyright 2012-2026 CNRS-UM LIRMM, CNRS-AIST JRL
#

cmake_minimum_required(VERSION 3.22)

project(Eigen3ToPython)
set(PROJECT_VERSION 1.0.7)

find_package(Eigen3 NO_CMAKE_PACKAGE_REGISTRY)
if(NOT Eigen3_FOUND)
  find_package(Eigen3 MODULE REQUIRED)
endif()
message("Found Eigen3: ${EIGEN3_VERSION_STRING}")
message("Eigen include directory: ${EIGEN3_INCLUDE_DIR}")

option(PYTHON_BINDING "Generate Python binding" ON)
set(PYTHON_BINDING_USER_INSTALL_DEFAULT ON)
option(PYTHON_BINDING_USER_INSTALL "Install the Python binding in user space" ${PYTHON_BINDING_USER_INSTALL_DEFAULT})
option(DISABLE_TESTS "Disable unit tests." OFF)

configure_file(setup.in.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

macro(ADD_BINDINGS PYTHON PYTHON_EXECUTABLE SOURCES)
  # Create the build directory
  set(SETUP_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${PYTHON}/$<CONFIGURATION>")
  if(DEFINED CMAKE_BUILD_TYPE)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PYTHON}/${CMAKE_BUILD_TYPE}")
  else()
    foreach(CFG ${CMAKE_CONFIGURATION_TYPES})
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PYTHON}/${CFG}")
    endforeach()
  endif()
  # Copy the configured setup script there
  file(GENERATE
    OUTPUT "${SETUP_LOCATION}/setup.py"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/setup.py"
  )
  # Target to build the bindings
  set(TARGET_NAME eigen-${PYTHON}-bindings)
  add_custom_target(${TARGET_NAME} ALL
    COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON_EXECUTABLE} setup.py build_ext --inplace
    COMMENT "Generating local Eigen ${PYTHON} bindings"
    DEPENDS ${SOURCES} SOURCES ${SOURCES}
  )
  # Copy sources
  set(I 0)
  foreach(SRC ${SOURCES})
    set(FILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}")
    set(FILE_OUT "${SETUP_LOCATION}/${SRC}")
    add_custom_target(copy-sources-${I}-${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILE_IN} ${FILE_OUT}
      DEPENDS ${FILE_IN}
    )
    add_dependencies(${TARGET_NAME} copy-sources-${I}-${TARGET_NAME})
    math(EXPR I "${I} + 1")
  endforeach()
  # Manual target to force regeneration
  add_custom_target(force-${TARGET_NAME}
    COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON_EXECUTABLE} setup.py build_ext --inplace --force
    COMMENT "Generating local Eigen ${PYTHON} bindings (forced)"
  )
  if(NOT ${DISABLE_TESTS})
    enable_testing()
    add_test(NAME test-${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON_EXECUTABLE} -m "pytest"
    )
  endif()
  # Install targets
  if(DEFINED PYTHON_DEB_ROOT)
    add_custom_target(install-${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON_EXECUTABLE} setup.py install --root=${PYTHON_DEB_ROOT} --install-layout=deb
      COMMENT "Install Eigen ${PYTHON} bindings (Debian layout)"
    )
  else()
    set(PIP_EXTRA_OPTIONS "")
    set(PIP_EXTRA_OPTIONS "--no-build-isolation")
    add_custom_target(install-${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON_EXECUTABLE} -m pip install . --upgrade ${PIP_EXTRA_OPTIONS}
      COMMENT "Install Eigen ${PYTHON} bindings"
    )
  endif()
  # Install requirements
  install(CODE "EXECUTE_PROCESS(COMMAND \"${CMAKE_COMMAND}\" --build \"${CMAKE_BINARY_DIR}\" --config \${CMAKE_INSTALL_CONFIG_NAME} --target install-${TARGET_NAME})")
endmacro()

set(BINDINGS_SOURCES
  eigen/c_eigen_private.pxd
  eigen/c_eigen.pxd
  include/eigen_wrapper.hpp
  utils/angleaxis.in.pyx
  utils/extras.in.pyx
  utils/generate_eigen_pyx.py
  utils/__init__.py
  utils/quaternion.in.pyx
  tests/test_eigen.py
)

if(${PYTHON_BINDING})
  if(NOT DEFINED Python3_EXECUTABLE)
    # Look for the python3 currently available in the PATH environment
    find_program(Python3_EXECUTABLE python3)

    if(NOT Python3_EXECUTABLE)
      message(FATAL_ERROR "Could not find python3 executable in your environment PATH.")
    else()
      message(STATUS "Found Python interpreter: ${Python3_EXECUTABLE}")
    endif()
  endif()

  ADD_BINDINGS(python3 "${Python3_EXECUTABLE}" "${BINDINGS_SOURCES}")
else()
  enable_testing()
  install(CODE "message(\"Nothing to install\")")
endif()
