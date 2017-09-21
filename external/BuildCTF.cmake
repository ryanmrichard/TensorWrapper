find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(CTF_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fPIC)
set(CTF_BLAS "${LAPACKE_LIBRARIES}")
string(REPLACE ";" " " CTF_FLAGS "${CTF_FLAGS}")
string(REPLACE ";" " " CTF_BLAS "${CTF_BLAS}")
set(CTF_ROOT ${CMAKE_BINARY_DIR}/external/CTF_external-prefix/src)
set(CTF_SRC_DIR ${CTF_ROOT}/CTF_external)
set(CTF_BUILD_DIR ${CTF_ROOT}/CTF_external-build)
set(CTF_INSTALL_DIR ${STAGE_DIR}/${CMAKE_INSTALL_PREFIX})
set(CTF_CONFIG_CMD ${CTF_SRC_DIR}/configure
                       CXX=${MPI_CXX_COMPILER}
                       CXXFLAGS=${CTF_FLAGS}
                       --blas=${CTF_BLAS}
                       --build-dir=${CTF_BUILD_DIR}
)
set(CTF_HEADER ${CTF_INSTALL_DIR}/include/ctf/ctf.hpp)
ExternalProject_Add(CTF_external
    GIT_REPOSITORY https://github.com/solomonik/ctf.git
    UPDATE_COMMAND ${CMAKE_COMMAND} -E echo "No update"
    PATCH_COMMAND ${CMAKE_COMMAND} -E echo "No patch"
    BINARY_DIR ${CTF_BUILD_DIR}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E echo "No configure"
    BUILD_COMMAND ${CTF_CONFIG_CMD} && ${CMAKE_MAKE_PROGRAM}
    BUILD_ALWAYS 0
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CTF_SRC_DIR}/include/ctf.hpp
                                             ${CTF_INSTALL_DIR}/include/ctf/sub_dir/ctf.hpp
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CTF_SRC_DIR}/src
                                             ${CTF_INSTALL_DIR}/include/ctf/src
            COMMAND ${CMAKE_COMMAND} -E copy ${CTF_BUILD_DIR}/lib/libctf.a
                                             ${CTF_INSTALL_DIR}/lib/libctf.a
            COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/write_ctf_header.sh
                            ${CTF_BUILD_DIR}/config.mk
                            ${CTF_HEADER}
)
