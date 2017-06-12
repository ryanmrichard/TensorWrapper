set(GA_PREFIX ${CMAKE_BINARY_DIR}/external/GlobalArrays_external-prefix)
set(GA_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ExternalProject_Add(GlobalArrays_external
    GIT_REPOSITORY https://github.com/GlobalArrays/ga.git
    GIT_TAG v5.6
    UPDATE_COMMAND ./autogen.sh
    CONFIGURE_COMMAND ${GA_PREFIX}/src/GlobalArrays_external/configure
                      CXX=${MPI_CXX_COMPILER}
                      CC=${MPI_C_COMPILER}
                      CXXFLAGS=${GA_FLAGS}
                      CFLAGS=${GA_FLAGS}
                      --with-pic
                      --prefix=${STAGE_DIR}/${CMAKE_INSTALL_PREFIX}
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
)
