#include <TensorWrapper/TensorWrapper.hpp>
#include <iostream>
#include "TestHelpers.hpp"

using namespace TWrapper;
using tensor_type=EigenTensor<2,double>;
using wrapped_type=tensor_type::wrapped_t;
using idx_t=Eigen::IndexPair<int>;
template<size_t n>
using idx_array=std::array<idx_t,n>;
int main()
{
    Tester tester("Testing Eigen Tensor Wrapping");
    const size_t dim=10;
    detail_::TensorConverter<2,double,detail_::TensorTypes::EigenTensor,
            detail_::TensorTypes::EigenMatrix> converter;
    Eigen::MatrixXd __A=Eigen::MatrixXd::Random(dim,dim),
                    __B=Eigen::MatrixXd::Random(dim,dim),
                    __C=Eigen::MatrixXd::Random(dim,dim);
    wrapped_type A=converter.convert(__A),
                 B=converter.convert(__B),
                 C=converter.convert(__C);
    tensor_type Default;
    Default=tensor_type(std::array<size_t,2>{dim,dim});
    tensor_type _A(A),_B(B),_C(C);
    wrapped_type D=A+B+C;
    tensor_type _D=_A+_B;//+_C;
    tester.test("Addition",_D==D);
//    Shape<2> corr_shape(std::array<size_t,2>{dim,dim},false);
//    tester.test("Tensors are same",_D==D);
//    //Test all local memory
//    {
//        auto memory=_D.get_memory();
//        tester.test("Memory shape",memory.local_shape==corr_shape);
//        tester.test("Memory get",memory(3,3)==D(3,3));
//        memory(3,3)=999.9;
//        Default.set_slice(memory);
//    }
//    tester.test("Memory set",_D.tensor()(3,3)==999.9);
//    tester.test("Set slice",Default.tensor()(3,3)==999.9);

//    std::array<size_t,2> start{2,1},end{3,3};
//    auto slice=_D.slice(start,end);
//    Shape<2> temp({1,2},false);
//    tester.test("Memory shape",slice.dims()==temp);
//    tester.test("Memory get",slice(0ul,0ul)==D(2,1));
//    tester.test("Element access",_D({2ul,2ul})==D(2,2));


//    wrapped_type E=A-B-C;
//    tensor_type _E=_A-_B-_C;
//    tester.test("Subtraction",_E==E);

//    wrapped_type F=E*0.5;//Eigen::Tensor doesn't overload the lhs...
//    tensor_type _F=0.5*_E;
//    tester.test("Double on lhs",_F==F);

//    wrapped_type G=E*0.5;
//    tensor_type _G=_E*0.5;
//    tester.test("Double on rhs",_G==G);

//    wrapped_type H=G.contract(A,idx_array<1>({idx_t({1,0})}));
//    tensor_type _H=_G("i,k")*_A("k,j");
//    tester.test("Contraction",_H==H);

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
