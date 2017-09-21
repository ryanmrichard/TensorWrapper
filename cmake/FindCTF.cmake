find_package(MPI REQUIRED)
find_path(CTF_INCLUDE_DIR ctf/ctf.hpp)
find_library(CTF_LIBRARY NAMES ctf)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CTF DEFAULT_MSG CTF_INCLUDE_DIR CTF_LIBRARY)
mark_as_advanced(CTF_LIBRARY CTF_INCLUDE_DIR)
set(CTF_LIBRARIES ${CTF_LIBRARY} ${MPI_CXX_LIBRARIES})
set(CTF_INCLUDE_DIRS ${CTF_INCLUDE_DIR} ${MPI_CXX_INCLUDE_PATH})
