#include <TensorWrapper/TensorWrapper.hpp>
#include <iostream>
#include "TestHelpers.hpp"

//using namespace TWrapper;
//using tensor_type=TensorWrapper<2,double,detail_::TensorTypes::GlobalArrays>;
//using wrapped_type=GATensor<2,double>;

int main()
{
    Tester tester("Testing Global Arrays Matrix Wrapping");
//    GAInitialize();
//    const size_t dim=10;
//    const std::array<size_t,2> dims({dim,dim});
//    const std::array<size_t,2> zeros({0,0});
//    //Abuse Eigen to get random matrices
//    Eigen::MatrixXd __A=Eigen::MatrixXd::Random(dim,dim),
//                 __B=Eigen::MatrixXd::Random(dim,dim),
//                 __C=Eigen::MatrixXd::Random(dim,dim);
//    //Convert to GA
//    GATensor<2,double> A(dims),B(dims),C(dims);
//    A.set_values(zeros,dims,__A.transpose().data());
//    B.set_values(zeros,dims,__B.transpose().data());
//    C.set_values(zeros,dims,__C.transpose().data());

//    tensor_type _A(A),_B(B),_C(C);
//    wrapped_type D=A+B+C;
//    tensor_type _D=_A+_B+_C;
//    Shape<2> corr_shape(dims,true);
//    tester.test("Tensor dimensions are correct",_D.dims()==corr_shape);
//    tester.test("Tensors are same",_D==D);

//    wrapped_type E=A-B-C;
//    tensor_type _E=_A-_B-_C;
//    tester.test("Tensor dimensions are correct",_E.dims()==corr_shape);
//    tester.test("Tensors are same",_E==E);

//    wrapped_type F=0.5*E;
//    tensor_type _F=0.5*_E;
//    tester.test("Tensor dimensions are correct",_F.dims()==corr_shape);
//    tester.test("Tensors are same",_F==F);

//    wrapped_type G=E*0.5;
//    tensor_type _G=_E*0.5;
//    tester.test("Tensor dimensions are correct",_G.dims()==corr_shape);
//    tester.test("Tensors are same",_G==G);

//    wrapped_type H=G*E;
//    tensor_type _H=_G("i,j")*_E("j,k");
//    tester.test("G * E",H==_H);

//    wrapped_type I=G.transpose()*E;
//    tensor_type _I=_G("j,i")*_E("j,k");
//    tester.test("G^T * E",I==_I);

//    wrapped_type J=G*E.transpose();
//    tensor_type _J=_G("i,j")*_E("k,j");
//    tester.test("G * E^T",J==_J);

//    wrapped_type K=G.transpose()*E.transpose();
//    tensor_type _K=_G("j,i")*_E("k,j");
//    tester.test("G^T * E^T",K==_K);

//    Eigen::MatrixXd I(2,2);
//    I<<1, 2, 2, 3;
//    TensorWrapper<2,double,detail_::TensorTypes::EigenMatrix> __I(I);
//    auto values=self_adjoint_eigen_solver(__I);
//    tensor_type _I(__I);
//    auto eigen_sys=self_adjoint_eigen_solver(_I);
//    tester.test("Eigenvalues",eigen_sys.first==values.first);
//    tester.test("Eigenvectors",eigen_sys.second==values.second);

    return tester.results();
}
