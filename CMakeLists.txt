CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/python.cmake)

SET(PROJECT_NAME Eigen3ToPython)
SET(PROJECT_DESCRIPTION "Bidirectional bridge between NumPy (Python 2) and Eigen 3.")
SET(PROJECT_URL "https://github.com/jorisv/Eigen3ToPython")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++0x -pedantic")

SETUP_PROJECT()

# Look for Python 2.7
SET(Python_ADDITIONAL_VERSIONS 2.7)
FINDPYTHON()

# Use Eigen 3
SET(Eigen_REQUIRED "eigen3 >= 3.2.0")
SEARCH_FOR_EIGEN()

ADD_SUBDIRECTORY(src)

SETUP_PROJECT_FINALIZE()
