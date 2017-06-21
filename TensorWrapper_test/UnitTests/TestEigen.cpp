#include <TensorWrapper/TensorWrapper.hpp>
#include <iostream>
#include "TestHelpers.hpp"

//using namespace TWrapper;
//template<size_t rank>
//using tensor_type=TensorWrapper<rank,double,detail_::TensorTypes::EigenMatrix>;
//using matrix_type=tensor_type<2>::wrapped_t;
//using vector_type=tensor_type<1>::wrapped_t;

int main()
{
    Tester tester("Testing Eigen Matrix Wrapping");
//    const size_t dim=10;
//    matrix_type A=matrix_type::Random(dim,dim),
//                 B=matrix_type::Random(dim,dim),
//                 C=matrix_type::Random(dim,dim);
//    Shape<2> corr_shape({10,10},false);

//    //Make sure a default constructed instance is available
//    tensor_type<2> Default;
//    tester.test("Default Dimensions",Default.dims()==Shape<2>({0,0},false));
//    Default=tensor_type<2>(std::array<size_t,2>{dim,dim});

//    //Ensure construction from Eigen Matrix
//    tensor_type<2> _A(A),_B(B),_C(C);

//    //Addition
//    matrix_type D=A+B+C;
//    tensor_type<2> _D=_A+_B+_C;
//    tester.test("Addition",_D==D);

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

//    matrix_type F=0.5*E;
//    tensor_type<2> _F=0.5*_E;
//    tester.test("Left scale",_F==F);

//    matrix_type G=E*0.5;
//    tensor_type<2> _G=_E*0.5;
//    tester.test("Right scale",_G==G);

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
