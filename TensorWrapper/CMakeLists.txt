cmake_minimum_required(VERSION 3.2)
project(TensorWrapper-Core CXX)
include(${TWRAPPER_ROOT}/cmake/cmake_macros.cmake)

#Header-only so nothing to do besides...

# Install all headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION include
        FILES_MATCHING
        REGEX ".*\.h(pp)?$")

# Write the configure file so other packages can find us
configure_file("${TWRAPPER_ROOT}/cmake/TensorWrapperConfig.cmake.in"
               TensorWrapperConfig.cmake @ONLY)

set(CBLAS_INCLUDE_FILE_SET "0")
set(LAPACKE_INCLUDE_FILE_SET "0")

if(CBLAS_INCLUDE_FILE)
    set(CBLAS_INCLUDE_FILE_SET "1")
endif()

if(LAPACKE_INCLUDE_FILE)
    set(LAPACKE_INCLUDE_FILE_SET "1")
endif()

configure_file("MathLibs.hpp.in" MathLibs.hpp @ONLY)

# Install said files
install(FILES ${CMAKE_BINARY_DIR}/TensorWrapperConfig.cmake
        DESTINATION share/cmake/TensorWrapper)

install(FILES ${CMAKE_BINARY_DIR}/MathLibs.hpp
        DESTINATION include/TensorWrapper)
