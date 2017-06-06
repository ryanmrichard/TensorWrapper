#include <TensorWrapper/TensorWrapper.hpp>
#include <iostream>
#include "TestHelpers.hpp"

using namespace TWrapper;
using tensor_type=TensorWrapper<2,double,Eigen::Tensor<double,2>>;
using wrapped_type=tensor_type::wrapped_t;
using idx_t=Eigen::IndexPair<int>;
template<size_t n>
using idx_array=std::array<idx_t,n>;
int main()
{
    Tester tester("Testing Eigen Tensor Wrapping");
    const size_t dim=10;
    detail_::TensorConverter<wrapped_type> convert;
    Eigen::MatrixXd __A=Eigen::MatrixXd::Random(dim,dim),
                    __B=Eigen::MatrixXd::Random(dim,dim),
                    __C=Eigen::MatrixXd::Random(dim,dim);
    wrapped_type A=convert(__A),
                 B=convert(__B),
                 C=convert(__C);
    tensor_type _A(A),_B(B),_C(C);
    wrapped_type D=A+B+C;
    tensor_type _D=_A+_B+_C;
    tester.test("Addition",_D==D);

    wrapped_type E=A-B-C;
    tensor_type _E=_A-_B-_C;
    tester.test("Subtraction",_E==E);

    wrapped_type F=E*0.5;//Eigen::Tensor doesn't overload the lhs...
    tensor_type _F=0.5*_E;
    tester.test("Double on lhs",_F==F);

    wrapped_type G=E*0.5;
    tensor_type _G=_E*0.5;
    tester.test("Double on rhs",_G==G);

    wrapped_type H=G.contract(A,idx_array<1>({idx_t({1,0})}));
    tensor_type _H=_G("i,k")*_A("k,j");
    tester.test("Contraction",_H==H);

    return tester.results();
}
