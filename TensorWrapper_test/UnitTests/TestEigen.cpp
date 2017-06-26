#include <TensorWrapper/TensorWrapper.hpp>
#include <TensorWrapper/Operation.hpp>
#include<TensorWrapper/TensorImpl/TensorTypes.hpp>
#include <iostream>
#include "TestHelpers.hpp"

using namespace TWrapper;
using tensor_type=EigenMatrix<double>;
using matrix_type=Eigen::MatrixXd;
using vector_type=Eigen::VectorXd;

int main()
{
    Tester tester("Testing Eigen Matrix Wrapping");
    const size_t dim=10;
    const std::array<size_t,2> shape({10,10});
    matrix_type A=matrix_type::Random(dim,dim),
                B=matrix_type::Random(dim,dim),
                C=matrix_type::Random(dim,dim);
    Shape<2> corr_shape(shape,false);

    //Basic Constructors
    tensor_type defaulted;
    tester.test("Default rank",defaulted.rank()==2);
    tensor_type allocated(shape);
    tester.test("Allocate rank",allocated.rank()==2);
    tester.test("Allocate dimensions",allocated.shape()==corr_shape);
    tensor_type moved(std::move(allocated));
    tester.test("Moved rank",moved.rank()==2);
    tester.test("Moved dimensions",moved.shape()==corr_shape);

    //Basic Assignment
    allocated=moved;
    tester.test("Assignment rank",allocated.rank()==2);
    tester.test("Assignment dimensions",allocated.shape()==corr_shape);
    defaulted=std::move(moved);
    tester.test("Move allocate rank",defaulted.rank()==2);
    tester.test("Allocate dimensions",defaulted.shape()==corr_shape);


    //Construction from Eigen Matrix
    tensor_type _A(A),_B(B),_C(C);
    tester.test("Rank",_A.rank()==2);
    tester.test("Dimensions",_A.shape()==corr_shape);
    tester.test("Values",_A==A);

    matrix_type D=A+B+C;
    tensor_type _D=_A+_B+_C;
    tester.test("Addition",_D==D);

    matrix_type E=0.5*D;
    tensor_type _E=0.5*_D;
    tester.test("Left scale",_E==E);

    matrix_type F=D*0.5;
    tensor_type _F=_D*0.5;
    tester.test("Right scale",_F==F);


//    //Test memory read/write
//    {
//        auto memory=_D.get_memory();
//        tester.test("Memory shape",memory.local_shape==corr_shape);
//        tester.test("Memory get",memory(3,3)==D(3,3));
//        memory(3,3)=999.9;
//        Default.set_slice(memory);
//    }//Memory gets set here

//    tester.test("Memory set",_D.tensor()(3,3)==999.9);
//    tester.test("Set slice",Default.tensor()(3,3)==999.9);

//    //Test slice interface
//    auto slice=_D.slice({2,1},{3,3});
//    tester.test("Memory shape",slice.dims()==Shape<2>({1,2},false));
//    tester.test("Memory get",slice({0,0})==D(2,1));
//    tester.test("Element access",_D({2,2})==D(2,2));

//    matrix_type E=A-B-C;
//    tensor_type<2> _E=_A-_B-_C;
//    tester.test("Subtraction",_E==E);



//    matrix_type H=G*E;
//    tensor_type<2> _H=_G("i,j")*_E("j,k");
//    tester.test("G * E",H==_H);

//    matrix_type I=G.transpose()*E;
//    tensor_type<2> _I=_G("j,i")*_E("j,k");
//    tester.test("G^T * E",I==_I);

//    matrix_type J=G*E.transpose();
//    tensor_type<2> _J=_G("i,j")*_E("k,j");
//    tester.test("G * E^T",J==_J);

//    matrix_type K=G.transpose()*E.transpose();
//    tensor_type<2> _K=_G("j,i")*_E("k,j");
//    tester.test("G^T * E^T",K==_K);

//    //Self-adjoint Eigen solver
//    matrix_type L(2,2);
//    L<<1, 2, 2, 3;
//    Eigen::SelfAdjointEigenSolver<matrix_type> solver(L);
//    tensor_type<2> _L(L);
//    auto eigen_sys=self_adjoint_eigen_solver(_L);
//    const Shape<1> vec_shape({2},false);

//    tester.test("Vector dims",eigen_sys.first.dims()==vec_shape);
//    tester.test("Eigenvalues",eigen_sys.first==solver.eigenvalues());
//    tester.test("Eigenvectors",eigen_sys.second==solver.eigenvectors());

    return tester.results();
}
