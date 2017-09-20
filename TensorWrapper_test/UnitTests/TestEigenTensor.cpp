#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace TWrapper::detail_;
using eigen_tensor=Eigen::Tensor<double,2>;
using idx_t=Eigen::IndexPair<int>;
template<size_t n>
using idx_array=std::array<idx_t,n>;

int main()
{
    Tester tester("Testing Eigen Tensor Wrapping");
    const size_t dim=10;
    const std::array<size_t,2> dims{dim,dim};
    TensorWrapperImpl<2,double,TensorTypes::EigenTensor> impl;

    eigen_tensor A(dim,dim);
    A.setConstant(3.0);

    //Shape
    Shape<2> corr_shape(dims,false);
    tester.test("Shape",impl.dims(A)==corr_shape);

    //Get/Set memory
    auto mem=impl.get_memory(A);
    tester.test("NBlocks",mem.nblocks()==1);
    tester.test("Memory contents",mem.block(0)==A.data());

    eigen_tensor B=impl.allocate(dims);
    impl.set_memory(B,mem);
    auto mem2=impl.get_memory(B);
    tester.test("Set memory",
                std::equal(A.data(),A.data()+100,mem2.block(0)));

    //Equality
    tester.test("Are equal",impl.are_equal(A,A));

    //TATensor F,G;
    //F("i,j")=B("i,j").block({0,0},{5,5});
    //std::cout<<F<<std::endl;
    //G=impl.slice(B,{4,4},{6,6});
    //mem=impl.get_memory(G);
    //std::array<double,4> corr_slice({3.0,3.0,4.0,3.0});
    //tester.test("Slice",std::equal(corr_slice.begin(),corr_slice.end(),
    //                               mem.block(0));


    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");
    //using idx_ii=detail_::make_indices<decltype(i),decltype(i)>;
    using idx_ij=detail_::make_indices<decltype(i),decltype(j)>;
    using idx_ik=detail_::make_indices<decltype(i),decltype(k)>;
    using idx_ji=detail_::make_indices<decltype(j),decltype(i)>;
    using idx_jk=detail_::make_indices<decltype(j),decltype(k)>;
    using idx_kl=detail_::make_indices<decltype(k),decltype(l)>;

    //Addition
    eigen_tensor E,D,C=A+B;
    D=impl.add<idx_ij,idx_ij>(A,B);
    tester.test("A + B",impl.are_equal(C,D));
    D=A+B+C;
    E=impl.add<idx_ij,idx_ij>(impl.add<idx_ij,idx_ij>(A,B),C);
    tester.test("A + B + C",impl.are_equal(D,E));
    C=A+B.shuffle(std::array<int,2>{1,0});
    D=impl.add<idx_ij,idx_ji>(A,B);
    tester.test("A + B^T",impl.are_equal(C,D));

    //Subtraction
    C=A-B;
    D=impl.subtract<idx_ij,idx_ij>(A,B);
    tester.test("A - B",impl.are_equal(C,D));
    D=A-B-C;
    E=impl.subtract<idx_ij,idx_ij>(impl.subtract<idx_ij,idx_ij>(A,B),C);
    tester.test("A - B - C",impl.are_equal(D,E));
    C=A-B.shuffle(std::array<int,2>{1,0});
    D=impl.subtract<idx_ij,idx_ji>(A,B);
    tester.test("A - B^T",impl.are_equal(C,D));

    //Scale
    C=A*0.5;
    D=impl.scale<idx_ij>(A,0.5);
    tester.test("Scale",impl.are_equal(C,D));

    //Permute
    C=B.shuffle(std::array<int,2>{1,0});
    D=impl.permute(B,{1,0});
    tester.test("Permutation",impl.are_equal(C,D));

    //Trace
    //Eigen::Tensor<double,0> x=A.trace(std::array<int,2>{0,1});
    //Eigen::Tensor<double,0> y=impl.trace<idx_ii>(A);
    //tester.test("Trace of A",impl.are_equal(x,y));

    //Contraction
    D=A.contract(B,idx_array<1>{idx_t{1,0}});
    E=impl.contraction<idx_ij,idx_jk>(A,B);
    tester.test("A * B",impl.are_equal(D,E));
    D=A.contract(B,idx_array<1>{idx_t{0,0}});
    E=impl.contraction<idx_ji,idx_jk>(A,B);
    tester.test("A^T * B",impl.are_equal(D,E));
    D=A.contract(B,idx_array<1>{idx_t{0,0}}).
            contract(C,idx_array<1>{idx_t{1,0}});
    E=impl.contraction<idx_ik,idx_kl>(impl.contraction<idx_ji,idx_jk>(A,B),C);
    tester.test("A^T * B * C",impl.are_equal(D,E));

    //Eval
    C=A+B;
    E=impl.eval<idx_ij>(impl.add<idx_ij,idx_ij>(A,B),dims);
    tester.test("Eval",impl.are_equal(C,E));


    Eigen::MatrixXd I(2,2);
    I<<1, 2, 2, 3;
    eigen_tensor _I(2,2);
    _I(0,0)=1;_I(0,1)=2;_I(1,0)=2;_I(1,1)=3;
    auto values=impl.self_adjoint_eigen_solver(_I);
    auto eigen_sys=Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(I);
    tester.test("Eigenvalues",eigen_sys.eigenvalues()(0)==values.first(0)&&
                              eigen_sys.eigenvalues()(1)==values.first(1));
    tester.test("Eigenvectors",
                eigen_sys.eigenvectors()(0,0)==values.second(0,0)&&
                eigen_sys.eigenvectors()(0,1)==values.second(0,1)&&
                eigen_sys.eigenvectors()(1,0)==values.second(1,0)&&
                eigen_sys.eigenvectors()(1,1)==values.second(1,1));

    return tester.results();
}
