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

\code{.cpp}
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
\endcode

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

\code{.cpp}
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

auto i=make_index("i");
auto j=make_index("j");
auto k=make_index("k");

//Usual matrix multiplication
TWrapper::EigenMatrix<2,double> product=A(i,k)*B(k,j);

\endcode

Similar to many of the underlying backends, TensorWrapper relies on lazy
evaluation to provide the above user-friendly API, while still allowing the
results to be highly optimized.  What this means to you the user is the
following:

\code{.cpp}
//A will actually contain the result of adding B and C
TWrapper::EigenMatrix<2,double> A=B+C;

//_A does not contain the result, but rather is a very thin object that
//describes the details of how to add B and C
auto _A=B+C;
\endcode

The latter line of code allows the user to write equations all on one line like:
\code{.cpp}
TWrapper::EigenMatrix<2,double> result=A(i,k)*B(k,j)-A(i,k)*D(k,j);
\endcode

and have the backend (possibly) optimize it at compile-time (whether it does or
doesn't depends on the backend; backends like Eigen will perform some

\code{.cpp}
//Instead of C*A+C*B do C*(A+B)
TWrapper::EigenMatrix<2,double> R1=A+B;
TWrapper::EigenMatrix<2,double> result=C(i,k)*R1(k,j);
\endcode

or






