#This file doesn't really look for CBLAS/LAPACKE it just makes sure the
#variables are set.

include(FindPackageHandleStandardArgs)
set(HAVE_MKL FALSE)
find_path(MKL_HEADER_PATH mkl.h
          PATHS ${LAPACKE_INCLUDE_DIR}
          NO_DEFAULT_PATH
)
if(MKL_HEADER_PATH)
    set(HAVE_MKL TRUE)
    set(LAPACKE_INCLUDE_FILE "mkl.h")
    set(CBLAS_FOUND TRUE)
else()
   find_path(CBLAS_HEADER_PATH cblas.h
             PATHS ${CBLAS_INCLUDE_DIR}
             NO_DEFAULT_PATH
             )
   if(CBLAS_HEADER_PATH)
        set(CBLAS_INCLUDE_FILE "cblas.h")
   else()
        find_path(CBLAS_HEADER_PATH blas.h
                  PATHS ${CBLAS_INCLUDE_DIR}
                  NO_DEFAULT_PATH
                  )
              if(CBLAS_HEADER_PATH)
                  set(CBLAS_INCLUDE_FILE "blas.h")
              endif()
   endif()
   find_path(LAPACKE_HEADER_PATH lapacke.h
             PATHS ${LAPACKE_INCLUDE_DIR}
             NO_DEFAULT_PATH
             )
  if(LAPACKE_HEADER_PATH)
      set(LAPACKE_INCLUDE_FILE "lapacke.h")
  endif()
  find_package_handle_standard_args(CBLAS
          REQUIRED_VARS CBLAS_INCLUDE_DIR
                        CBLAS_LIBRARIES
                        CBLAS_INCLUDE_FILE
          )
   message(STATUS "Using BLAS libraries: ${CBLAS_LIBRARIES}")
   message(STATUS "Using BLAS includes: ${CBLAS_INCLUDE_DIR}")
   message(STATUS "BLAS header file: ${CBLAS_INCLUDE_FILE}")
endif()

find_package_handle_standard_args(LAPACKE
           REQUIRED_VARS LAPACKE_INCLUDE_DIR
                         LAPACKE_LIBRARIES
                         LAPACKE_INCLUDE_FILE
)

find_package_handle_standard_args(CBLASLAPACKE
           REQUIRED_VARS CBLAS_FOUND LAPACKE_FOUND
)

message(STATUS "Using LAPACK libraries: ${LAPACKE_LIBRARIES}")
message(STATUS "Using LAPACK includes: ${LAPACKE_INCLUDE_DIR}")
message(STATUS "LAPACKe header file: ${LAPACKE_INCLUDE_FILE}")
