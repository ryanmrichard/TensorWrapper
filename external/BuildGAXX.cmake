set(GA_PREFIX ${CMAKE_BINARY_DIR}/external/GlobalArraysTemp_external-prefix)
enable_language(Fortran)
find_package(MPI REQUIRED)
ExternalProject_Add(GAXX_external
    GIT_REPOSITORY https://github.com/ryanmrichard/ga
    GIT_TAG ga_cxx
    UPDATE_COMMAND echo "No ppdate"
    PATCH_COMMAND echo "No patch"
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_Fortran_COMILER=${CMAKE_Fortran_COMPILER}
               -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
               -DMPI_C_COMPILER=${MPI_C_COMPILER}
               -DMPI_Fortran_COMPILER=${MPI_Fortran_COMPILER}
               -DCMAKE_INSTALL_PREFIX=${STAGE_DIR}/${CMAKE_INSTALL_PREFIX}
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
    CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:LIST=${CMAKE_PREFIX_PATH}
                     -DCMAKE_INSTALL_RPATH:LIST=${CMAKE_INSTALL_RPATH}
                     -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
                     -DCMAKE_MODULE_PATH:LIST=${CMAKE_MODULE_PATH}
)

