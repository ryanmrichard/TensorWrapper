#include <TensorWrapper/TensorWrapper.hpp>
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

    tensor_type _A({dim,dim}),_B({dim,dim}),_C({dim,dim});
    FillRandom(_A);
    FillRandom(_B);
    FillRandom(_C);
    wrapped_type A=_A.data(),B=_B.data(),C=_C.data();

    Shape<2> corr_shape(std::array<size_t,2>{dim,dim},false);
    tester.test("Allocated shape",_A.shape()==corr_shape);

    auto memory=_A.get_memory();
    tester.test("Memory shape",memory.local_shape==corr_shape);
    tester.test("Memory get",memory(3,3)==A({3,3}));
    memory(3,3)=999.9;
    _A.set_memory(memory);
    tester.test("Memory set",_A(3,3)==999.9);
    A({3,3})=999.9;

    tensor_type slice=_A.slice({2,1},{3,3});
    tester.test("Slice shape",slice.shape()==Shape<2>({1,2},false));
    tester.test("Same element",slice(0,0)==A(2,1));

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
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    tensor_type _H=_G(i,k)*_A(k,j);
    tester.test("Contraction",_H==H);

    Eigen::MatrixXd I(2,2);
    I<<1, 2, 2, 3;
    EigenMatrix<double> __I(I);
    auto values=self_adjoint_eigen_solver(__I);
    tensor_type _I(__I);
    auto eigen_sys=self_adjoint_eigen_solver(_I);
    tester.test("Eigenvalues",eigen_sys.first==EigenTensor<1,double>(values.first));
    tester.test("Eigenvectors",eigen_sys.second==tensor_type(values.second));

    return tester.results();
}
