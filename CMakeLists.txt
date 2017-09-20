cmake_minimum_required(VERSION 3.1)
project(TensorWrapper-superbuild C CXX)
include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/cmake_macros.cmake)

############################
# Options
############################

#Should we build the stress tests?
option_w_default(BUILD_TESTS TRUE)
#Debug or Release build?
option_w_default(CMAKE_BUILD_TYPE "Release")

################################################################################
# BLAS / LAPACK
################################################################################

find_package(CBLASLAPACKE REQUIRED)

################################################################################
# Backends
#
# Each backend has a file in external that corresponds to obtaining the source,
# configuring it, and compiling it (if applicable).  Each backend will then be
# found using the normal find_package methodology where they are expected to
# create a target with their name as listed in the XXX variable below.
#
################################################################################

#Enable Cyclops Tensor Framework
option_w_default(ENABLE_CTF FALSE)
#Enable Eigen backend?
#Warning: things probably won't work if you disable this one
option_w_default(ENABLE_Eigen3 TRUE)
#Enable Global Arrays backend?
option_w_default(ENABLE_GAXX FALSE)
#Enable Tiled Array backend?
option_w_default(ENABLE_tiledarray FALSE)

#List of all backends we know about.  Add new ones here.
set(XXX "CTF" "Eigen3" "GAXX" "tiledarray")

#BACKENDS is the sublist of XXX that is only the ones that are turned on.
foreach(x ${XXX})
    if(${ENABLE_${x}})
        set(BACKENDS_ON ${BACKENDS_ON} ${x})
        message(STATUS "${x} backend is enabled.")
    else()
        set(BACKENDS_OFF ${BACKENDS_OFF} ${x})
        message(STATUS "${x} backend is disabled")
    endif()
endforeach()

##################################################
# Where stuff will be installed. This is passed
# to other external projects
##################################################
# Note that CMAKE_INSTALL_PREFIX is always absolute
# (cmake makes it that way). Therefore it always begins
# with a slash
set(STAGE_DIR            "${CMAKE_BINARY_DIR}/stage")
set(STAGE_INSTALL_PREFIX "${STAGE_DIR}${CMAKE_INSTALL_PREFIX}")

# Where targets should search for dependencies
# Ie, the original CMAKE_PREFIX_PATH, plus the installation stage
list(APPEND CMAKE_PREFIX_PATH ${STAGE_INSTALL_PREFIX})

add_subdirectory(external)

#####################################################
#  Main Build
#####################################################

ExternalProject_Add(TensorWrapper
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/TensorWrapper
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DTWRAPPER_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
               -DCBLAS_INCLUDE_DIR=${CBLAS_INCLUDE_DIR}
               -DCBLAS_INCLUDE_FILE=${CBLAS_INCLUDE_FILE}
               -DLAPACKE_INCLUDE_DIR=${LAPACKE_INCLUDE_DIR}
               -DLAPACKE_INCLUDE_FILE=${LAPACKE_INCLUDE_FILE}
               -DHAVE_MKL=${HAVE_MKL}
    BUILD_ALWAYS 1
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:LIST=${CMAKE_PREFIX_PATH}
                     -DCMAKE_INSTALL_RPATH:LIST=${CMAKE_INSTALL_RPATH}
                     -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
                     -DCMAKE_MODULE_PATH:LIST=${CMAKE_MODULE_PATH}
                     -DCBLAS_LIBRARIES:LIST=${CBLAS_LIBRARIES}
                     -DLAPACKE_LIBRARIES:LIST=${LAPACKE_LIBRARIES}
                     -DBACKENDS_ON:LIST=${BACKENDS_ON}
                     -DBACKENDS_OFF:LIST=${BACKENDS_OFF}
)

foreach(x ${BACKENDS_ON})
    add_dependencies(TensorWrapper ${x}_external)
endforeach()

###############################
# Tests
###############################
if(${BUILD_TESTS})
    set(TENSORWRAPPER_PREFIX ${STAGE_DIR}${CMAKE_INSTALL_PREFIX}/TensorWrapper)
    ExternalProject_Add(TensorWrapper_test
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/TensorWrapper_test
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/test_stage
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                   -DSTAGE_DIR=${STAGE_DIR}
                   -DTENSORWRAPPER_PREFIX=${TENSORWRAPPER_PREFIX}
                   -DENABLE_STRESS_TESTS=${ENABLE_STRESS_TESTS}
        BUILD_ALWAYS 1
        CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:LIST=${CMAKE_PREFIX_PATH}
                         -DCMAKE_INSTALL_RPATH:LIST=${CMAKE_INSTALL_RPATH}
                         -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
                         -DCMAKE_MODULE_PATH:LIST=${CMAKE_MODULE_PATH}
    )
    add_dependencies(TensorWrapper_test TensorWrapper)
endif()


# This file will allow us to run ctest in the top-level build dir
# Basically it just defers to the actual top-level CTestTestfile.cmake in the
# build directory for this project
file(WRITE ${CMAKE_BINARY_DIR}/CTestTestfile.cmake "subdirs(test_stage)")

# Install the staging directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_PREFIX}/
        DESTINATION ${CMAKE_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)

#Install our header files
install(DIRECTORY "cmake"
        DESTINATION share/cmake
        FILES_MATCHING REGEX ".*\.cmake")

#####################
# uninstall target
#####################
configure_file(
"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
IMMEDIATE @ONLY)

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
