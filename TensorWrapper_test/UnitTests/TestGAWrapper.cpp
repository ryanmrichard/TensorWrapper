#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"
using namespace TWrapper;
using namespace detail_;

int main()
{
    Tester tester("Testing Global Arrays Matrix Wrapping");
#ifdef ENABLE_GAXX
    GAInitialize();
    const size_t dim=10;
    using tensor_type=TensorWrapper<2,double,detail_::TensorTypes::GlobalArrays>;
    using wrapped_type=GATensor<2,double>;
    detail_::TensorWrapperImpl<2,double,detail_::TensorTypes::GlobalArrays> impl;

    using index_t=std::array<size_t,2>;
    const index_t dims({dim,dim}), zeros({0,0});
    tensor_type _A(dims),_B(dims),_C(dims);
    fill_random(_A);
    fill_random(_B);
    fill_random(_C);

    wrapped_type A=_A.data(),B=_B.data(),C=_C.data();

    //Dimensions
    Shape<2> corr_shape(dims,true);
    tester.test("Matrix shape",impl.dims(A)==corr_shape);

    //Get Memory
    auto mem=impl.get_memory(A);
    tester.test("Matrix NBlocks",mem.nblocks()==1);
    auto vals=A.get_values(zeros,dims);
    tester.test("Matrix pointer",std::equal(mem.block(0),mem.block(0)+100,
                                            vals.begin()));

    //Set Memory
    mem.block(0)[0]=999.0;
    impl.set_memory(A,mem);
    double corr_elem=A.get_values(zeros,std::array<size_t,2>{1,1})[0];
    tester.test("Matrix Set Memory",corr_elem==999.0);

    //Slice
    wrapped_type slice=impl.slice(A,{2,1},{3,3});
    auto corr_vals=A.get_values(std::array<size_t,2>{2,1},
                      std::array<size_t,2>{3,3});
    vals=slice.get_values(zeros,std::array<size_t,2>{1,2});
    tester.test("Matrix slice",corr_vals==vals);

    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");
    using idx_i=make_indices<decltype(i)>;
    using idx_ii=make_indices<decltype(i),decltype(i)>;
    using idx_j=make_indices<decltype(j)>;
    using idx_ij=make_indices<decltype(i),decltype(j)>;
    using idx_ji=make_indices<decltype(j),decltype(i)>;
    using idx_jk=make_indices<decltype(j),decltype(k)>;
    using idx_ik=make_indices<decltype(i),decltype(k)>;
    using idx_kj=make_indices<decltype(k),decltype(j)>;
    using idx_kl=make_indices<decltype(k),decltype(l)>;

    //Addition
    wrapped_type D=A+B;
    wrapped_type E=impl.add<idx_ij,idx_ij>(A,B);
    tester.test("Matrix A+B",D==E);
    D=A+B+C;
    E=impl.add<idx_ij,idx_ij>(impl.add<idx_ij,idx_ij>(A,B),C);
    tester.test("Matrix A+B+C",D==E);
    D=A+B.transpose();
    E=impl.add<idx_ij,idx_ji>(A,B);
    tester.test("Matrix A+B^T",D==E);


    //Subtraction
    D=A-B;
    E=impl.subtract<idx_ij,idx_ij>(A,B);
    tester.test("Matrix A-B",D==E);
    D=A-B-C;
    E=impl.subtract<idx_ij,idx_ij>(impl.subtract<idx_ij,idx_ij>(A,B),C);
    tester.test("Matrix A-B-C",D==E);
    D=A-B.transpose();
    E=impl.subtract<idx_ij,idx_ji>(A,B);
    tester.test("Matrix A-B^T",D==E);

    //Scaling
    D=A*0.5;
    E=impl.scale<idx_ij>(A,0.5);
    tester.test("Matrix scale",D==E);

    //Trace
    double x=A.trace();
    double y=impl.trace<idx_ii>(A);
    tester.test("Trace of A",x==y);

    //Contraction
    D=A*B;
    E=impl.contraction<idx_ij,idx_jk>(A,B);
    tester.test("Matrix A * B",D==E);
    D=A*B.transpose();
    E=impl.contraction<idx_ij,idx_kj>(A,B);
    tester.test("Matrix A * B^T",D==E);
    D=A.transpose()*B;
    E=impl.contraction<idx_ji,idx_jk>(A,B);
    tester.test("Matrix A^T * B",D==E);
    D=A.transpose()*B.transpose();
    E=impl.contraction<idx_ji,idx_kj>(A,B);
    tester.test("Matrix A^T * B^T",D==E);
    D=A.transpose()*B*C;
    E=impl.contraction<idx_ik,idx_kl>(impl.contraction<idx_ji,idx_jk>(A,B),C);
    tester.test("Matrix A^T * B * C",D==E);
    double corr_s=A.dot(B);
    double s=impl.contraction<idx_ij,idx_ij>(A,B);
    tester.test("A(i,j) * B(i,j)",corr_s==s);
    corr_s=A.dot(B.transpose());
    s=impl.contraction<idx_ij,idx_ji>(A,B);
    tester.test("A(i,j) * B(j,i)",corr_s==s);


//    //Self-adjoint Eigen solver
//    eigen_matrix L(2,2);
//    L<<1, 2, 2, 3;
//    Eigen::SelfAdjointEigenSolver<eigen_matrix> solver(L);
//    auto eigen_sys=impl.self_adjoint_eigen_solver(L);

//    tester.test("Eigenvalues",eigen_sys.first==solver.eigenvalues());
//    tester.test("Eigenvectors",eigen_sys.second==solver.eigenvectors());

#endif
    return tester.results();
}
