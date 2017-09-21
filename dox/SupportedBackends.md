Supported Backends
==================

At the moment TensorWrapper supports the following backends subject to the
caveats laid out in each section.

Cyclops Tensor Framework
------------------------

Project homepage is [here](https://github.com/solomonik/ctf)

Cyclops Tensor Framework (CTF) is both a shared and distributed memory tensor
library.  CTF's claim to fame is its use of a cyclic data structure to store
the tensor (*i.e.* the first element of the tensor goes to process 0, the second
to process 1, the third to process 2, *etc.*).  Admittedly other tensor
libraries support this structure, but it is usually a somewhat buried option as
opposed to the default.

### Notes on the Wrapping

- By default CTF does not support installing and expects you to use the code out
  of the build directory.  Unfortunately, the build directory is a mess in that
  it includes object files and other assorted odds and ends that don't belong in
  an installation.  We recommend you let TensorWrapper build CTF so that the
  result is installable.
- CTF is not const correct with member functions of the tensor class (either
  that or expressions like: `C["ij"]=A["ij"]+B["ij"]` really do modify A and B).
  The result is a lot of `const_cast`'s are needed in the wrapper.  We point
  this out as a disclaimer because if the `const_cast`'s aren't valid you are
  likely to get some weird errors (based on our .

Eigen
-----

Project homepage is [here](https://bitbucket.org/eigen/eigen/)

Support for both Eigen's vector/matrix library and it's (somewhat hidden) tensor
library are included.  This backend is a shared memory implementation that comes
with its own BLAS/LAPACK kernels or can be made to use already existing, highly
optimized BLAS/LAPACK libraries.  By default Eigen's vector/matrix library and
tensor libraries are not fully interchangable (*i.e.* it is not in general
possible to mix tensors and and matrices); however, TensorWrapper allows this
limitation to be circumvented.

### Notes on the Wrapping

- As wrapped by TensorWrapper, when applicable, Eigen's internal BLAS/LAPACK
kernels are always substituted for standard BLAS/LAPACK kernels.
- Eigen tensor uses C++11 threads for threading.  To make the threading interface
uniform to the user TensorWrapper will initialize the threadpool, when needed,
with the current number of available OpenMP threads.
-Eigen's tensor library does not include support for eigen decomposition.
For rank2 Eigen tensors, TensorWrapper calls the standard Eigen matrix library
routines.


Global Arrays
-------------

Project homepage is [here](http://hpc.pnl.gov/globalarrays/), GitHub repository
is [here](https://github.com/GlobalArrays/ga).

Although the API supports arbitrary numbers of indices, Global Arrays really
only contains support for matrix operations.  This backend is a distributed
implementation that is used in several electronic structure packages including
NWChem, Molpro, and Psi4.  Out of the box it only supports very basic matrix
routines (add, subtract, scale, multiply, etc.) and does not include support for
contraction.

Tiled Array
-----------

Project homepage is[here](https://github.com/ValeevGroup/tiledarray).

Tiled Array is both a shared and distributed memory tensor library.
