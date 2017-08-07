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

# Install said file
install(FILES ${CMAKE_BINARY_DIR}/TensorWrapperConfig.cmake
        DESTINATION share/cmake/TensorWrapper)
