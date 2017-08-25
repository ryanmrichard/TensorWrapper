Quick Start   {#QuickStart}
===========

The point of this page is to get you quickly up and running with the
TensorWrapper library.  Since TensorWrapper does not provide any of its own
linear algebra features it will need to know what backend to start with.  For
the point of this tutorial we will assume you have choosen Eigen, but since
the API is largely agnostic to the backend it should be trivial for the reader
to extend thes examples to other backends.

Declaring and Filling a Tensor
------------------------------

Let's start by making a tensor.

~~~cpp
//This includes the TensorWrapper library
#include<TensorWrapper/TensorWrapper.hpp>

int main()
{
    //A default constructed matrix of doubles
    TWrapper::EigenMatrix<2,double> matrix1;

    //A default constructed vector of floats
    TWrapper::EigenMatrix<1,float> vector1;

    //A 10 by 10 matrix of doubles
    TWrapper::EigenMatrix<2,double> matrix2({10,10});

    //A 5 element long vector of doubles using the native Eigen type.
    Eigen::VectorXd vector2(5);

    //This is one way to fill an Eigen::VectorXd
    vector2<<1.1, 2.2, 3.3, 4.4, 5.5;

    //Wraps an already existing native instance
    TWrapper::EigenMatrix<1,std::complex<double>> vector3(vector2);

    return 0;
}//End main function
~~~

The code is useless in the sense that it doesn't do anything, but it does
demonstrate the basics of making a tensor with the TensorWrapper library.  Note
that usage of the `TensorWrapper/TensorWrapper.hpp` header file brings in to
scope all of the functionality of the TensorWrapper library.  All of this
functionality is namespace protected by the `TWrapper` namespace.  This example
shows off the three constructors we expect users to explicitly invoke:

- Default constructor: Used to make the `matrix1` and `vector1` instances above.
    It basically constructs a place-holder instance that can be assigned to
    later.
- Allocate: Used to make `matrix2` instance.  This will allocate
    the memory for the tensor.  The exact details of where that memory is
    located, how it is arranged, etc. are determined by the backend.
- Initialize from a native instance: Used to make `vector3` instance.  This
    constructor is the one we anticpate users using the most.  Basically it will
    wrap an existing instance of the backend (by default this is done via copy,
    but one can also use the move version to allow TensorWrapper to take
    ownership of the instance).

TensorWrapper does have a mechanism for generically filling the tensors that
we omit from this quick start.  Instead we recommend users make and fill an
instance of the backend and then provide TensorWrapper with that instance.
Finally, we note that the usual copy/move and copy assign/move assign operators
are defined and available.

Linear Algebra
-----------------

After making and filling a tensor the next thing you'll likely want to do is
use it.  The following demonstrates how to do basic operations with
TensorWrapper:

~~~cpp
//We assume this code takes place inside a function

//We assume you already initialized and filled these
TWrapper::EigenMatrix<2,double> A,B,C;

//Addition
TWrapper::EigenMatrix<2,double> sum=A+B+C;

//Subtraction
TWrapper::EigenMatrix<2,double> difference=A-B-C;

//Left multiply by a scalar
TWrapper::EigenMatrix<2,double> Ax2=2.0*A;

//Right multiply by a scalar
TWrapper::EigenMatrix<2,double> Aover2=A*0.5;

//Declare some indices, trust me you want to use auto as the resulting type is
//nasty...
auto i=make_index("i");
auto j=make_index("j");
auto k=make_index("k");

//Usual matrix multiplication
TWrapper::EigenMatrix<2,double> product=A(i,k)*B(k,j);

//A transpose times B
TWrapper::EigenMatrix<2,double> product2=A(k,i)*B(k,j);
~~~

Similar to many of the underlying backends, TensorWrapper relies on lazy
evaluation to provide the above user-friendly API, while still allowing the
results to be highly optimized.  What this means to you the user is the
following:

~~~cpp
//A will actually contain the result of adding B and C
TWrapper::EigenMatrix<2,double> A=B+C;

//_A does not contain the result, but rather is a very thin object that
//describes the details of how to add B and C and has a horrific looking type
auto _A=B+C;
~~~

Basically, unless you assign the result to a TensorWrapper instance your
resulting object is some unevaluated part of the equation.

Backend Agnosticism
-------------------

The examples above all use the Eigen backend and via the `TWrapper::EigenMatrix`
type this decision is hard-coded.  One of the major selling points of
TensorWrapper is the fact that we can write generic routines, regardless of
which tensor library we want to use.  To do this, one must write their routines
to accept instances of the common base class, `TWrapper::TensorWrapperBase`.
This type is templated on the element type and the rank only.  What this means
is that say you have a function, which for simplicity we
assume is just matrix multiplication, to capitalize on the backend agnosticism
of TensorWrapper you simply need to write your algorithm like this:

~~~.cpp
TWrapper::TensorWrapperBase<2,double> my_algorithm(
    const TWrapper::TensorWrapperBase<2,double>& matrix1,
    const Twrapper::TensorWrapperBase<2,double>& matrix2)
{
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");

    TWrapper::EigenMatrix<2,double> A(matrix1),B(matrix2);

    return A(i,k)*B(k,j);
}
~~~

Note that we still had to select a backend to actually do the evaluation (the
best you can do for a completely backend free code is to template the function
on the backend type), but our algorithm now works with input tensors coming from
any backend.  The initialization of the `A` and `B` instances in the above code
takes care of any required conversions (or is basically a pointer dereference if
`matrix1` and `matrix2` already are Eigen matrices).

As a slight technical aside, the reason we can't simply do
`matrix1(i,k)*matrix2(k,j)` is because they may actually have different
backends.  In that case TensorWrapper would have to choose which backend to use
or be able to do its own linear algebra in terms of TensorWrapperBase's API.
The first option is non-transparent to the user and the second one requires
TensorWrapper to write its own optimized linear algebra kernels (thereby
defeating the purpose of using backends to begin with).







