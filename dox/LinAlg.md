Linear Algebra Features {#LinAlg}
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
to the variables `i`,`j`, and `k`.

## Addition

As you likely expect, addition is invoked via the `+` sign:
```.cpp
//C_ij = A_ij + B_ij
TensorWrapperBase<2,T> C=A(i,j)+B(i,j);

//C_ij = A_ij + B_ij
C=A+B;

//Can't mix indexed and non-indexed notations
//C=A+B(i,j);  //<----This won't compile

//C_ij = A_ij + B_ji
C=A(i,j)+B(j,i);
```

For basic operations TensorWrapper supports either an indexed form or a
non-indexed form.
