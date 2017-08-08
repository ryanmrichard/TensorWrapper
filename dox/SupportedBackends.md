Supported Backends {#backends}
==================

At the moment TensorWrapper supports the following backends

Eigen
-----

Support for both Eigen's vector/matrix library and it's (somewhat hidden) tensor
library are included.  This backend is a shared memory implementation that comes
with its own BLAS/LAPACK kernels or can be made to use already existing, highly
optimized BLAS/LAPACK libraries.  By default Eigen's vector/matrix library and
tensor libraries are not fully interchangable (*i.e.* it is not in general
possible to mix tensors and and matrices); however, TensorWrapper allows this
limitation to be circumvented.

:note: Eigen's tensor library does not include support for eigen decomposition.
For rank2 Eigen tensors, TensorWrapper calls the standard matrix library
routines.

Global Arrays
-------------

Although the API supports arbitrary numbers of indices, Global Arrays really
only contains support for matrix operations.  This backend is a distributed
implementation that is used in several electronic structure packages including
NWChem, Molpro, and Psi4.  Out of the box it only supports very basic matrix
routines (add, subtract, scale, multiply, etc.) and does not include support for
