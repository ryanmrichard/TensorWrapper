cmake_minimum_required(VERSION 3.2)
project(TensorWrapper-Core CXX)
include(${TWRAPPER_ROOT}/cmake/cmake_macros.cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS False)

# CMake doesn't support Intel CXX standard until cmake 3.6
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
  if("${CMAKE_VERSION}" VERSION_LESS "3.6")
      add_compile_options(-std=c++${CMAKE_CXX_STANDARD})
  endif()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(TENSORWRAPPER_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
set(TENSORWRAPPER_LIBRARIES ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

foreach(pkg ${BACKENDS_ON})
    set(ENABLE_${pkg} 1)
    find_package(${pkg} REQUIRED)
    STRING(TOUPPER ${pkg} PKG)
    set(TENSORWRAPPER_LIBRARIES ${TENSORWRAPPER_LIBRARIES}
                                     ${${PKG}_LIBRARIES})
    set(TENSORWRAPPER_INCLUDE_DIRS ${TENSORWRAPPER_INCLUDE_DIRS}
                                   ${${PKG}_INCLUDE_DIRS})
endforeach()

foreach(pkg ${BACKENDS_OFF})
    set(ENABLE_${pkg} 0)
endforeach()

c_ify(HAVE_MKL HAVE_MKL)
c_ify(MPI_CXX_FOUND HAVE_MPI)
c_ify(BUILD_LIBRARY BUILDING_LIBRARY)

set(CBLAS_INCLUDE_FILE_SET "0")
if(CBLAS_INCLUDE_FILE)
    set(CBLAS_INCLUDE_FILE_SET "1")
endif()

set(LAPACKE_INCLUDE_FILE_SET "0")
if(LAPACKE_INCLUDE_FILE)
    set(LAPACKE_INCLUDE_FILE_SET "1")
endif()


configure_file("MathLibs.hpp.in" MathLibs.hpp @ONLY)
configure_file("Config.hpp.in" TensorWrapper/Config.hpp @ONLY)

if(${BUILD_LIBRARY})
    add_library(TensorWrapper SHARED IndexItr.cpp
                                     MemoryBlock.cpp
                                     Shape.cpp
                                     TensorPtr.cpp
                                     TensorWrapperExtern.cpp

    )
    target_include_directories(TensorWrapper PRIVATE
                                             ${TWRAPPER_ROOT}
                                             ${TENSORWRAPPER_INCLUDE_DIRS}
                                             ${CBLAS_INCLUDE_DIR}
                                             ${LAPACKE_INCLUDE_DIR}
                                             ${CMAKE_BINARY_DIR}
    )
    target_compile_options(TensorWrapper PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(TensorWrapper PRIVATE ${TENSORWRAPPER_LIBRARIES})
    install(TARGETS TensorWrapper DESTINATION lib/TensorWrapper)
endif()

# Write the configure file so other packages can find us
configure_file("${TWRAPPER_ROOT}/cmake/TensorWrapperConfig.cmake.in"
               TensorWrapperConfig.cmake @ONLY)

# Install all headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION include
        FILES_MATCHING
        REGEX ".*\.h(pp)?$")

# Install said files
install(FILES ${CMAKE_BINARY_DIR}/TensorWrapperConfig.cmake
        DESTINATION share/cmake/TensorWrapper)

install(FILES ${CMAKE_BINARY_DIR}/MathLibs.hpp
        DESTINATION include/TensorWrapper)

install(FILES ${CMAKE_BINARY_DIR}/TensorWrapper/Config.hpp
        DESTINATION include/TensorWrapper)
