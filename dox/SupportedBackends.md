Supported Backends {#backends}
==================

At the moment TensorWrapper supports the following backends subject to the
caveats laid out in each section.

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
