# - Try to find Global Arrays
# Once done this will define
#  GlobalArrays_FOUND - System has Global Arrays
#  GlobalArrays_INCLUDE_DIRS - The Global Arrays include directories
#  GlobalArrays_LIBRARIES - The libraries needed to use Global Arrays

find_path(GLOBALARRAYS_GA_INCLUDE_DIR ga.h
          HINTS ${PC_GLOBALARRAYS_INCLUDEDIR} ${PC_GLOBALARRAYS_INCLUDE_DIRS}
         )
find_path(GLOBALARRAYS_MA_INCLUDE_DIR macdecls.h
          HINTS ${PC_GLOBALARRAYS_INCLUDEDIR} ${PC_GLOBALARRAYS_INCLUDE_DIRS}
)

find_library(GLOBALARRAYS_C_LIBRARY NAMES libga${CMAKE_STATIC_LIBRARY_SUFFIX}
             HINTS ${PC_GLOBALARRAYS_LIBDIR} ${PC_GLOBALARRAYS_LIBRARY_DIRS} )

find_library(GLOBALARRAYS_ARMCI_LIBRARY libarmci${CMAKE_STATIC_LIBRARY_SUFFIX}
             HINTS ${PC_GLOBALARRAYS_LIBDIR} ${PC_GLOBALARRAYS_LIBRARY_DIRS} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GlobalArrays DEFAULT_MSG
                                  GLOBALARRAYS_C_LIBRARY
                                  GLOBALARRAYS_ARMCI_LIBRARY
                                  GLOBALARRAYS_GA_INCLUDE_DIR
                                  GLOBALARRAYS_MA_INCLUDE_DIR
)

mark_as_advanced(GLOBALARRAYS_INCLUDE_DIR GLOBALARRAYS_LIBRARY )

set(GlobalArrays_LIBRARIES ${GLOBALARRAYS_C_LIBRARY}
                           ${GLOBALARRAYS_ARMCI_LIBRARY}
)
set(GlobalArrays_INCLUDE_DIRS ${GLOBALARRAYS_GA_INCLUDE_DIR}
                              ${GLOBALARRAYS_MA_INCLUDE_DIR}
)
