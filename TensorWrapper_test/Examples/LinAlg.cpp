#include <TensorWrapper/TensorWrapper.hpp>

/** \file This file contains the code examples found in the LinAlg
 *  documentation.  If this test breaks, those have broken too...
 */

int main()
{

    auto mu=make_indices("mu");
    auto nu=make_indices("nu");
    auto dont_do_this=make_indices("mu");
    static_assert(dont_do_this==mu,"This assert will pass");

    //Example assumes the next few lines were included
    auto i=make_indices("i");
    auto j=make_indices("j");
    auto k=make_indices("k");

    TensorWrapper::EigenMatrix<double> A(Eigen::MatrixXd::Random(10,10)),
                                       B(Eigen::MatrixXd::Random(10,10)),
                                       C;

    //C_ij = A_ij + B_ij
    C=A(i,j)+B(i,j);

    //C_ij = A_ij + B_ij
    C=A+B;

    //Can't mix indexed and non-indexed notations
    //C=A+B(i,j);  //<----This won't compile

    //C_ij = A_ij + B_ji
    C=A(i,j)+B(j,i);

    //C_ij = A_ij - B_ij
    C=A(i,j)-B(i,j);

    //C_ij = A_ij - B_ij
    C=A-B;

    //Can't mix indexed and non-indexed notations
    //C=A-B(i,j);  //<----This won't compile

    //C_ij = A_ij - B_ji
    C=A(i,j)-B(j,i);

    //Left multiply by a scalar
    C=0.5*A(i,j);

    //Can be done without the indices
    C=0.5*A;

    //Scalar can appear on the right
    C=A(i,j)*0.5;

    //Scalar can appear on right without indices
    C=A*0.5;

    //Normal matrix multiplication
    C=A(i,j)*B(j,k);

    //A times B transpose
    C=A(i,j)*B(k,j);

    TensorWrapper::EigenScalar<double> D;

    //D is assumed to be a rank 0 tensor containing a double
    D=A(i,i);

    return 0;
}
