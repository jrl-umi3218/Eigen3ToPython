# Create the package in build dir for testing purpose
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/eigen3)
CONFIGURE_FILE(__init__.py ${CMAKE_CURRENT_BINARY_DIR}/eigen3/__init__.py COPYONLY)
CONFIGURE_FILE(convert.py ${CMAKE_CURRENT_BINARY_DIR}/eigen3/convert.py COPYONLY)
SET(OUTPUT_BINDING ${CMAKE_CURRENT_BINARY_DIR}/eigen3.cpp)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # The bindings cannot compile with clang and the c++11 flag, it would be
  # complicated to parse the generated C++ to patch it on the fly, hence this
  # not-so-elegant solution
  ADD_CUSTOM_COMMAND (
    OUTPUT ${OUTPUT_BINDING}
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/.clang/eigen3.cpp.clang ${OUTPUT_BINDING}
  )
else()
  # Generate Python binding code
  ADD_CUSTOM_COMMAND (
    OUTPUT ${OUTPUT_BINDING}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${OUTPUT_BINDING}
    DEPENDS generate.py
  )
endif()


# Build the library
SET(SOURCES ${OUTPUT_BINDING})
SET(HEADERS MatrixBaseAddon.h MatrixAddon.h EigenUtils.h)

SET(MatrixBaseAddon "\"${CMAKE_CURRENT_SOURCE_DIR}/MatrixBaseAddon.h\"")
SET(MatrixAddon "\"${CMAKE_CURRENT_SOURCE_DIR}/MatrixAddon.h\"")
ADD_DEFINITIONS(-DEIGEN_MATRIXBASE_PLUGIN=${MatrixBaseAddon}
                -DEIGEN_MATRIX_PLUGIN=${MatrixAddon})

ADD_LIBRARY(_eigen3 MODULE ${SOURCES} ${HEADERS})
target_include_directories(_eigen3 SYSTEM PUBLIC ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
target_include_directories(_eigen3 BEFORE PRIVATE  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_link_libraries(_eigen3 PUBLIC ${PYTHON_LIBRARY})

SET_TARGET_PROPERTIES(_eigen3 PROPERTIES PREFIX "")

# Install rules

# Targets:
#   * <prefix>/lib/libbar.a
#   * <prefix>/lib/libbaz.a
#   * header location after install: <prefix>/include/foo/Bar.hpp
#   * headers can be included by C++ code `#include <foo/Bar.hpp>`
install(
    TARGETS _eigen3
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "${PYTHON_SITELIB}/eigen3"
    ARCHIVE DESTINATION "${PYTHON_SITELIB}/eigen3"
    RUNTIME DESTINATION "${PYTHON_SITELIB}/eigen3"
    INCLUDES DESTINATION "${INCLUDE_INSTALL_DIR}"
)
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DESTINATION})
PYTHON_INSTALL_BUILD(eigen3 __init__.py "${PYTHON_SITELIB}")
PYTHON_INSTALL_BUILD(eigen3 convert.py "${PYTHON_SITELIB}")

