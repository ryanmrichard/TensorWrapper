Configuring and Compiling {#Building}
=========================

We have done our best to make the configuration and compilation of TensorWrapper
as painless as possible.  This page contains our collective wisdom for when the
standard process:

~~~.sh
cmake -H. -Bbuild
make
make install
~~~

doesn't work.

CMake Basics
------------

Before discussing the more specific configure/build options/problems it makes
sense to list some basics of CMake.  First, options are passed to
CMake using the syntax `-DOPTION_NAME=VALUE`.  If you need to set an option to
multiple values (such as options controlling search paths) you will need to put
the values in double quotes and seperate the values with semicolons, *i.e.* try
`-DOPTION_WITH_TWO_VALUES="VALUE1;VALUE2"`.

CMake by default already includes a whole host of options for configuring common
build settings.  Ones that are particularly relevant to this project are
included in the following table along with a brief description of what they do.

--------------------------------------------------------------------------------
| CMake Variable | Description                                                 |
| :------------: | :-----------------------------------------------------------|
| CMAKE_CXX_COMPILER | The C++ compiler that will be used                      |
| CMAKE_CXX_FLAGS | Flags that will be passed to C++ compiler                  |
| MPI_CXX_COMPILER | MPI C++ wrapper compiler (should wrap CMAKE_CXX_COMPILER) |
| CMAKE_C_COMPILER | The C compiler that will be used                          |
| CMAKE_C_FLAGS | Flags that will be passed to C compiler                      |
| MPI_C_COMPILER | MPI C wrapper compiler (should wrap CMAKE_C_COMPILER)       |
| CMAKE_Fortran_COMPILER | The Fortran compiler that will be used              |
| CMAKE_Fortran_FLAGS | Flags that will be passed to the Fortran compiler      |
| MPI_Fortran_COMPILER | MPI Fortran wrapper compiler (should wrap CMAKE_Fortran_COMPILER) |
| CMAKE_BUILD_TYPE | Debug, Release, or RelWithDebInfo                         |
| CMAKE_PREFIX_PATH | A list of places CMake will look for dependencies        |
| CMAKE_INSTALL_PREFIX | The install directory                                 |
--------------------------------------------------------------------------------

A general piece of advice, when working with CMake always pass full paths.


Math Libraries
--------------

This is a math-based library so suffice it to say we need BLAS/LAPACK.
Unfortunately there does not seem to be any good BLAS/LAPACK cmake detection
modules out there so you'll have to set the values yourself.  TensorWrapper
requires "modern" BLAS/LAPACK so-called CBLAS/LAPACKE respectively.

-------------------------------------------------------------------------------
| Variable Name | Description                                                 |
| :-----------: | :-----------------------------------------------------------|
| CBLAS_LIBRARIES | The path(s) to the BLAS library(ies) you want to use      |
| CBLAS_INCLUDE_DIR | The path to the include directory for the BLAS header   |
| LAPACKE_LIBRARIES | The path(s) to the LAPACKE library(ies)                 |
| LAPACKE_INCLUDE_DIR | The path to the include directory for the LAPACKE header|
-------------------------------------------------------------------------------

If you are using MKL it is sufficient to simply set the LAPACKE versions of the
variables.  Whether you are using MKL or not will be detected based on whether
or not `mkl.h` is found in the path that `LAPACKE_INCLUDE_DIR` is set to.


