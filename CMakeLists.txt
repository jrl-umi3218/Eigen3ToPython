#
# Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
#

cmake_minimum_required(VERSION 2.8)

project(Eigen3ToPython)
set(PROJECT_VERSION 1.0.2)

find_package(Eigen3 NO_CMAKE_PACKAGE_REGISTRY)
if(NOT Eigen3_FOUND)
  find_package(Eigen3 MODULE REQUIRED)
endif()
message("Found Eigen3: ${EIGEN3_VERSION_STRING}")
message("Eigen include directory: ${EIGEN3_INCLUDE_DIR}")

option(PYTHON_BINDING "Generate Python binding" ON)
if(WIN32)
  set(PYTHON_BINDING_USER_INSTALL_DEFAULT ON)
else()
  set(PYTHON_BINDING_USER_INSTALL_DEFAULT OFF)
endif()
option(PYTHON_BINDING_USER_INSTALL "Install the Python binding in user space" ${PYTHON_BINDING_USER_INSTALL_DEFAULT})
option(PYTHON_BINDING_FORCE_PYTHON2 "Use pip2/python2 instead of pip/python" OFF)
option(PYTHON_BINDING_FORCE_PYTHON3 "Use pip3/python3 instead of pip/python" OFF)
set(PYTHON_BINDING_BUILD_PYTHON2_AND_PYTHON3_DEFAULT OFF)
if(DEFINED PYTHON_DEB_ROOT)
  set(PYTHON_BINDING_BUILD_PYTHON2_AND_PYTHON3_DEFAULT ON)
endif()
option(PYTHON_BINDING_BUILD_PYTHON2_AND_PYTHON3 "Build Python 2 and Python 3 bindings" ${PYTHON_BINDING_BUILD_PYTHON2_AND_PYTHON3_DEFAULT})
if(${PYTHON_BINDING_FORCE_PYTHON2} AND ${PYTHON_BINDING_FORCE_PYTHON3})
  message(FATAL_ERROR "Cannot enforce Python 2 and Python 3 at the same time")
endif()

option(DISABLE_TESTS "Disable unit tests." OFF)

# Check if pip install supports --system
macro(_pip_has_install_system PIP OUT)
  execute_process(COMMAND ${PIP} install --system
                  RESULT_VARIABLE ${OUT}
                  OUTPUT_QUIET ERROR_QUIET)
  if(${${OUT}} EQUAL 0)
    set(${OUT} True)
  else()
    set(${OUT} False)
  endif()
endmacro()

configure_file(setup.in.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

macro(ADD_BINDINGS PYTHON PIP SOURCES)
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
    COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON} setup.py build_ext --inplace
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
    COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON} setup.py build_ext --inplace --force
    COMMENT "Generating local Eigen ${PYTHON} bindings (forced)"
  )
  if(NOT ${DISABLE_TESTS})
    enable_testing()
    add_test(NAME test-${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON} -c "import nose; nose.run()"
    )
  endif()
  # Install targets
  if(DEFINED PYTHON_DEB_ROOT)
    add_custom_target(install-${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PYTHON} setup.py install --root=${PYTHON_DEB_ROOT} --install-layout=deb
      COMMENT "Install Eigen ${PYTHON} bindings (Debian layout)"
    )
  else()
    set(PIP_EXTRA_OPTIONS "")
    if(${PYTHON_BINDING_USER_INSTALL})
      set(PIP_EXTRA_OPTIONS "--user")
    endif()
    if(DEFINED PIP_INSTALL_PREFIX)
      _pip_has_install_system(${PIP} PIP_HAS_INSTALL_SYSTEM)
      execute_process(COMMAND ${PYTHON} -c "import sys; print(\"python{}.{}\".format(sys.version_info.major, sys.version_info.minor));" OUTPUT_VARIABLE PYTHON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(PIP_TARGET "${PIP_INSTALL_PREFIX}/lib/${PYTHON_VERSION}/site-packages/")
      set(PIP_EXTRA_OPTIONS --target "${PIP_TARGET}")
      if(${PIP_HAS_INSTALL_SYSTEM})
        set(PIP_EXTRA_OPTIONS --system ${PIP_EXTRA_OPTIONS})
      endif()
    endif()
    add_custom_target(install-${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND} -E chdir "${SETUP_LOCATION}" ${PIP} install . ${PIP_EXTRA_OPTIONS} --upgrade
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
  if(${PYTHON_BINDING_BUILD_PYTHON2_AND_PYTHON3})
    ADD_BINDINGS("python2" "pip2" "${BINDINGS_SOURCES}")
    ADD_BINDINGS("python3" "pip3" "${BINDINGS_SOURCES}")
  elseif(${PYTHON_BINDING_FORCE_PYTHON3})
    ADD_BINDINGS("python3" "pip3" "${BINDINGS_SOURCES}")
  elseif(${PYTHON_BINDING_FORCE_PYTHON2})
    ADD_BINDINGS("python2" "pip2" "${BINDINGS_SOURCES}")
  else()
    ADD_BINDINGS("python" "pip" "${BINDINGS_SOURCES}")
  endif()
else()
  enable_testing()
  install(CODE "message(\"Nothing to install\")")
endif()
