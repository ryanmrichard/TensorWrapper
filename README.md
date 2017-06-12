TensorWrapper
=============

At the moment there are several competing tensor libraries.  Consequentially,
each developer has their preferance.  Particularly in a communial code base like
that of computational chemistry, this makes it hard to use libraries written by
other developers if they use a different tensor library.  Additionally, all of
the tensor libraries have a slightly different syntax which means one has to
rewrite code for each library.  TensorWrapper is designed to provide a common
API to the various tensor libraries without impacting performance.

As with most libraries designed to provide common APIs to disparate packages,
TensorWrapper relies on wrapping existing implementations.  Particularly for
tensor libraries, this is a difficult task as many of the more cutting edge
libraries obtain their speed via expression templating.  TensorWrapper allows
the underlying implementation to use expression templating under the hood by
gratitous usage of the `auto` keyword and then evaluating the expression when
the user needs it (typically at assignment).  At the moment TensorWrapper does
not apply any optimizations of its own.

- [Supported Tensor Backends](#supported-tensor-backends)
- [Installing](#installing)
  - [Obtaining](#obtaining)
  - [Configuring](#configuring)
  - [Building and Installing](#building-and-installing)
  - [Testing](#testing)

Supported Tensor Backends
=========================

At the moment these are all of the supported tensor backends.

| Name | Project Hompage | Notes |
| ---  | --------------- | ----- |
| Eigen Matrix |:link:[Eigen Homepage](eigen.tuxfamily.org/) | The default matrix backend |
| Eigen Tensor |:link:[Eigen Homepage](eigen.tuxfamily.org/) | The default tensor backend |
| Global Arrays |:link:[Global Arrays Homepage](http://hpc.pnl.gov/globalarrays/) | Depends on MPI and Fortran dependency |

:memo: For backends lacking support for some feature mandated by the
TensorWrapper API we fall back to Eigen as it provides all the support we need.

Installing
==========

Obtaining
---------
The official repository for TensorWrapper is
:link:[here](https://github.com/ryanmrichard/TensorWrapper) and the source code
can be obtained via the usual git clone command:

~~~.sh
git clone https://github.com/ryanmrichard/TensorWrapper
~~~

This will download the source code from GitHub, create a new directory
`TensorWrapper`in your current directory, and place the source code in
`TensorWrapper`.

Configuring
-----------

TensorWrapper tries to comply with the typical CMake configuration process, to
that end a command along the lines of:

~~~.sh
cmake -H. -Bbuild -DOPTION_1=VALUE1 -DOPTION_2=VALUE2 ...
~~~

should be sufficient to configure TensorWrapper.  To this end here are some
helpful, standard, CMake options that can be used to fine tune the build.

--------------------------------------------------------------------------------
| CMake Variable | Description                                                 |
| :------------: | :-----------------------------------------------------------|
| CMAKE_CXX_COMPILER | The C++ compiler that will be used                      |
| CMAKE_CXX_FLAGS | Flags that will be passed to C++ compiler                  |
| CMAKE_BUILD_TYPE | Debug, Release, or RelWithDebInfo                         |
| CMAKE_PREFIX_PATH | A list of places CMake will look for dependencies        |
| CMAKE_INSTALL_PREFIX | The install directory                                 |
--------------------------------------------------------------------------------

Depending on which backends are enabled you may also benefit from setting some
of these options:

--------------------------------------------------------------------------------
| CMake Variable | Description                                                 |
| :------------: | :-----------------------------------------------------------|
| CMAKE_Fortran_COMPILER | The Fortran compiler that will be used              |
| CMAKE_Fortran_FLAGS | Flags that will be passed to the Fortran compiler      |
| MPI_Fortran_COMPILER | The MPI wrapper compiler for building Fortran executables |
| MPI_CXX_COMPILER | The MPI wrapper compiler for building C++ executables     |
--------------------------------------------------------------------------------

Finally, these are the project dependent options:

--------------------------------------------------------------------------------
| CMake Variable | Description                                                 |
| :------------: | :-----------------------------------------------------------|
| ENABLE_GA      | Should the Global Arrays backend be built? (Default: False) |
| ENBALE_STRESS_TESTS | Should stress tests be built and added to the testing pool? (Default: False) |
--------------------------------------------------------------------------------

With the default options TensorWrapper only depends on the Eigen matrix/tensor
libraries (they are bundled together) and is header-only.  Enabling other
backends may impart additional dependencies, which are described in the table
in the :link:[Supported Tensor Backends](#supported-tensor-backends) section.

Building and Installing
-----------------------

After the configuration step is successful, the rest of the build should be as
simple as:

~~~.sh
cd build && make
make install
~~~

Testing
-------

TensorWrapper includes two forms of tests, unit tests for ensuring everything
works and stress tests for ensuring TensorWrapper has a minimal overhead at
runtime.  The former compile and run very quickly, the latter compile quickly,
but will require rather large amounts of resources to run.  By default, only the
unit tests can be run.  To run the unit tests on a successful staged, but not
installed build (*i.e.* between the `make` and `make install` steps) simply run:

~~~.sh
ctest
~~~

in the build directory.  Stress tests can be enabled by setting the option
`ENABLE_STRESS_TESTS` to true at configuration time.
