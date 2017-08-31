#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"
using namespace TWrapper;


int main()
{
    Tester tester("Testing Global Arrays Matrix Wrapping");
#ifdef ENABLE_GAXX
    using tensor_type=TensorWrapper<2,double,detail_::TensorTypes::GlobalArrays>;
    using wrapped_type=GATensor<2,double>;
    GAInitialize();

    const size_t dim=10;
    using index_t=std::array<size_t,2>;
    const index_t dims({dim,dim}), zeros({0,0});
    tensor_type _A(dims),_B(dims),_C(dims);
    FillRandom(_A);
    FillRandom(_B);
    FillRandom(_C);

    wrapped_type A(_A.data()),B(_B.data()),C(_C.data());
    tensor_type __A(A);
    tester.test("Construct by value",_A==A);

//    auto memory=_A.get_memory();
//    const Shape<2> corr_shape(dims,true);
//    tester.test("Memory shape",memory.local_shape==corr_shape);
//    tester.test("Memory get",memory(3,3)==A.get_values(index_t{3,3}));
//    memory(index_t{3,3})=999.9;
//    _A.set_memory(memory);
//    tester.test("Memory set",_A(3,3)==999.9);
//    double temp=999.9;
//    A.set_values(index_t{3,3},index_t{4,4},&temp);

    tensor_type slice=_A.slice({2,1},{3,3});
    tester.test("Slice shape",slice.shape()==Shape<2>({1,2},true));
    tester.test("Same element",slice(0,0)==A.get_values(index_t{2,1}));

    wrapped_type D=A+B+C;
    tensor_type _D=_A+_B+_C;
    tester.test("Addition",_D==D);

    wrapped_type E=A-B-C;
    tensor_type _E=_A-_B-_C;
    tester.test("Subtraction",_E==E);

    wrapped_type F=0.5*E;
    tensor_type _F=0.5*_E;
    tester.test("Right scalar multiplication",_F==F);

    wrapped_type G=E*0.5;
    tensor_type _G=_E*0.5;
    tester.test("Left scalar multiplication",_G==G);

    wrapped_type H=G*E;
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    tensor_type _H=_G(i,k)*_E(k,j);
    tester.test("G * E",H==_H);

    wrapped_type I=G.transpose()*E;
    tensor_type _I=_G(k,i)*_E(k,j);
    tester.test("G^T * E",I==_I);

    wrapped_type J=G*E.transpose();
    tensor_type _J=_G(i,k)*_E(j,k);
    tester.test("G * E^T",J==_J);

    wrapped_type K=G.transpose()*E.transpose();
    tensor_type _K=_G(k,i)*_E(j,k);
    tester.test("G^T * E^T",K==_K);

//    Eigen::MatrixXd I(2,2);
//    I<<1, 2, 2, 3;
//    EigenMatrix<2,double> __I(I);
//    auto values=self_adjoint_eigen_solver(__I);
//    tensor_type _I(__I);
//    auto eigen_sys=self_adjoint_eigen_solver(_I);
//    tester.test("Eigenvalues",eigen_sys.first==values.first);
//    tester.test("Eigenvectors",eigen_sys.second==values.second);
#endif
    return tester.results();
}
