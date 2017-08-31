find_package(MPI REQUIRED)
#For the moment Eigen3 is a requirement if it ever becomes optional this logic
#will need reworked (note TiledArray does not seem to use the find_package
find_package(Eigen3 REQUIRED)
ExternalProject_Add(tiledarray_external
    GIT_REPOSITORY https://github.com/ValeevGroup/tiledarray
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
               -DMPI_C_COMPILER=${MPI_C_COMPILER}
               -DCMAKE_INSTALL_PREFIX=${STAGE_DIR}/${CMAKE_INSTALL_PREFIX}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
               -DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIRS}
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
    CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:LIST=${CMAKE_PREFIX_PATH}
                     -DCMAKE_INSTALL_RPATH:LIST=${CMAKE_INSTALL_RPATH}
                     -DBLAS_LIBRARIES:LIST=${CBLAS_LIBRARIES}
                     -DLAPACK_LIBRARIES:LIST=${LAPACKE_LIBRARIES}
                     -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
                     -DCMAKE_MODULE_PATH:LIST=${CMAKE_MODULE_PATH}
)
