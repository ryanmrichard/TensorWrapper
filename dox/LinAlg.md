Linear Algebra Features
=======================

The point of this page is to document the features of the TensorWrapper API and
to tell you how to use them.

Preliminaries
-------------

Much of the API uses Einstein notation.  This can be summarized as:

1. An index may be repeated at most twice within any product of tensors, *i.e.*
   the product \f$R_{ij}S_{jk}T_{kl}\f$ is fine, but \f$R_{ij}S_{jk}T_{kk}\f$ is
   not.
2. If an index is repeated in a product of tensors summation over that index is
   implied, *i.e.* \f$R_{ij}S_{jk}\f$ is the same as \f$\sum_{j}R_{ij}S_{jk}\f$.

In TensorWrapper indices are made from arbitrary strings, which are then
manipulated at compile time.  For this to occur you need to register your
strings.  To do this simply call the `make_indices` function (note this is the
only function that is part of the TensorWrapper library that is not in the
TensorWrapper namespace (if you're curious why it is because it's actually a
macro)):

```.cpp
#include<TensorWrapper/TensorWrapper.hpp>
auto mu=make_indices("mu");
auto nu=make_indices("nu");
auto dont_do_this=make_indices("mu");

static_assert(dont_do_this==mu,"This assert will pass");
```

The use of `auto` for the return type is not strictly necessary; however, the
resulting type of the indices is quite nasty and it is best to not concern one's
self with it.  Take note of the `static_assert`; regardless of what you call
the instance resulting from `make_indices` it is the string content of that
instance that is being utilized and not the variable name itself.

Basic Operations
================

This section assumes that you have included `TensorWrapper/TensorWrapper.hpp` in
your file and have created 3 distinct index instances, which you have assigned
to the variables `i`,`j`, and `k`.  It also assumes you have declared three
rank 2 tensors whose elements are either floats or doubles and whose dimensions
are compatible (*i.e.* the following operations are defined).  These tensors are
assumed to be assigned to the variables `A`, `B`, and `C`.

## Addition

As you likely expect, addition is invoked via the `+` sign:
```.cpp
//C_ij = A_ij + B_ij
C=A(i,j)+B(i,j);

//C_ij = A_ij + B_ij
C=A+B;

//Can't mix indexed and non-indexed notations
//C=A+B(i,j);  //<----This won't compile

//C_ij = A_ij + B_ji
C=A(i,j)+B(j,i);
```

## Subtraction

Subtraction support is the same as addition except it is invoked via the `-`
sign:

```.cpp
//C_ij = A_ij - B_ij
C=A(i,j)-B(i,j);

//C_ij = A_ij - B_ij
C=A-B;

//Can't mix indexed and non-indexed notations
//C=A-B(i,j);  //<----This won't compile

//C_ij = A_ij - B_ji
C=A(i,j)-B(j,i);
```

## Scaling

TensorWrapper supports both left and right multiplying by a scalar (assuming
that scalar is of the same type, or is convertible to, the type in your tensor):

```.cpp
//Left multiply by a scalar
C=0.5*A(i,j);

//Can be done without the indices
C=0.5*A;

//Scalar can appear on the right
C=A(i,j)*0.5;

//Scalar can appear on right without indices
C=A*0.5;
```

A few general notes:
    - These operations support both an indexed form and a non-indexed form.
      Mixing of the two in a single expression will result in an error.
    - These operations can be combined to form more complicated expressions
      (subject to the caveat that all terms are either indexed or non-indexed).

Contraction
-----------

Contraction in TensorWrapper is always specified via Einstein notation:

```.cpp
//Normal matrix multiplication
C=A(i,j)*B(j,k);

//A times B transpose
C=A(i,j)*B(k,j);
```

Einstein notation can also be used to denote a trace:

```.cpp
//D is assumed to be a rank 0 tensor containing a double
D=A(i,i);
```
