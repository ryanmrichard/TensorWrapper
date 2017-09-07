#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace TWrapper::detail_;
using eigen_matrix=Eigen::MatrixXd;
using eigen_vector=Eigen::VectorXd;
using eigen_scalar=Eigen::Matrix<double,1,1>;

template<size_t R>
using impl_t=TensorWrapperImpl<R,double,TensorTypes::EigenMatrix>;

int main()
{
    Tester tester("Testing wrapping of Eigen matrix library");
    const size_t dim=10;
    const std::array<size_t,2> shape({dim,dim});
    const std::array<size_t,1> vshape({dim});
    const std::array<size_t,0> sshape({});
    eigen_matrix A=eigen_matrix::Random(dim,dim),
                 B=eigen_matrix::Random(dim,dim),
                 C=eigen_matrix::Random(dim,dim);
    eigen_vector vA=eigen_vector::Random(dim),
                 vB=eigen_vector::Random(dim),
                 vC=eigen_vector::Random(dim);
    eigen_scalar sA=eigen_scalar::Random(1),
                 sB=eigen_scalar::Random(1),
                 sC=eigen_scalar::Random(1);
    impl_t<2> impl;
    impl_t<1> vimpl;
    impl_t<0> simpl;

    //Dimensions
    Shape<2> corr_shape(shape,false);
    Shape<1> vcorr_shape(vshape,false);
    Shape<0> scorr_shape(sshape,false);
    tester.test("Matrix shape",impl.dims(A)==corr_shape);
    tester.test("Vector shape",vimpl.dims(vA)==vcorr_shape);
    tester.test("Scalar shape",simpl.dims(sA)==scorr_shape);

    //Get Memory
    auto mem=impl.get_memory(A);
    tester.test("Matrix NBlocks",mem.nblocks()==1);
    tester.test("Matrix pointer",mem.block(0)==A.data());
    auto vmem=vimpl.get_memory(vA);
    tester.test("Vector NBlocks",vmem.nblocks()==1);
    tester.test("Vector pointer",vmem.block(0)==vA.data());
    auto smem=simpl.get_memory(sA);
    tester.test("Scalar NBlocks",smem.nblocks()==1);
    tester.test("Scalar pointer",smem.block(0)==sA.data());

    //Set Memory
    mem.block(0)[0]=999.0;
    impl.set_memory(A,mem);
    tester.test("Matrix Set Memory",A(0,0)==999.0);
    vmem.block(0)[0]=999.0;
    vimpl.set_memory(vA,vmem);
    tester.test("Vector Set Memory",vA(0,0)==999.0);
    smem.block(0)[0]=999.0;
    simpl.set_memory(sA,smem);
    tester.test("Scalar Set Memory",sA(0,0)==999.0);

    //Slice
    eigen_matrix slice=impl.slice(A,{2,1},{3,3});
    tester.test("Matrix slice",slice(0,0)==A(2,1));
    eigen_vector vslice=vimpl.slice(vA,{2},{3});
    tester.test("Vector slice",vslice(0)==vA(2));


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
    eigen_matrix D=A+B;
    eigen_matrix E=impl.add<idx_ij,idx_ij>(A,B);
    tester.test("Matrix A+B",D==E);
    D=A+B+C;
    E=impl.add<idx_ij,idx_ij>(impl.add<idx_ij,idx_ij>(A,B),C);
    tester.test("Matrix A+B+C",D==E);
    D=A+B.transpose();
    E=impl.add<idx_ij,idx_ji>(A,B);
    tester.test("Matrix A+B^T",D==E);
    eigen_vector vD=vA+vB;
    eigen_vector vE=vimpl.add<idx_i,idx_i>(vA,vB);
    tester.test("Vector A+B",vD==vE);
    vD=vA+vB+vC;
    vE=vimpl.add<idx_i,idx_i>(vimpl.add<idx_i,idx_i>(vA,vB),vC);
    tester.test("Vector A+B+C",vD==vE);
    eigen_scalar sD=sA+sB;
    eigen_scalar sE=simpl.add<idx_i,idx_i>(sA,sB);
    tester.test("Scalar A+B",sD==sE);
    sD=sA+sB+sC;
    sE=simpl.add<idx_i,idx_i>(simpl.add<idx_i,idx_i>(sA,sB),sC);
    tester.test("Vector A+B+C",vD==vE);

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
    vD=vA-vB;
    vE=vimpl.subtract<idx_i,idx_i>(vA,vB);
    tester.test("Vector A-B",vD==vE);
    vD=vA-vB-vC;
    vE=vimpl.subtract<idx_i,idx_i>(vimpl.subtract<idx_i,idx_i>(vA,vB),vC);
    tester.test("Vector A-B-C",vD==vE);
    sD=sA-sB;
    sE=simpl.subtract<idx_i,idx_i>(sA,sB);
    tester.test("Scalar A-B",sD==sE);
    sD=sA-sB-sC;
    sE=simpl.subtract<idx_i,idx_i>(simpl.subtract<idx_i,idx_i>(sA,sB),sC);
    tester.test("Vector A-B-C",vD==vE);

    //Scaling
    D=A*0.5;
    E=impl.scale<idx_ij>(A,0.5);
    tester.test("Matrix scale",D==E);
    vD=vA*0.5;
    vE=vimpl.scale<idx_i>(vA,0.5);
    tester.test("Vector scale",vD==vE);
    sD=sA*0.5;
    sE=simpl.scale<Indices<>>(sA,0.5);
    tester.test("Scalar scale",sD==sE);

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
    double corr_s=A.cwiseProduct(B).sum();
    double s=impl.contraction<idx_ij,idx_ij>(A,B);
    tester.test("A(i,j) * B(i,j)",corr_s==s);
    corr_s=A.cwiseProduct(B.transpose()).sum();
    s=impl.contraction<idx_ij,idx_ji>(A,B);
    tester.test("A(i,j) * B(j,i)",corr_s==s);
    sD=vA.transpose()*vB;
    sE=vimpl.contraction<idx_i,idx_i>(vA,vB);
    tester.test("Vector A^T * B",sD==sE);
    D=vA*vB.transpose();
    E=vimpl.contraction<idx_i,idx_j>(vA,vB);
    tester.test("Vector A * B^T", D==E);
    D=vA.transpose()*B;
    E=vimpl.contraction<idx_i,idx_ij>(vA,B);
    tester.test("A(i) * B(i,j)",D==E);
    D=vA.transpose()*B.transpose();
    E=vimpl.contraction<idx_i,idx_ji>(vA,B);
    tester.test("A(i) * B(j,i)",D==E);
    D=A*vB;
    E=impl.contraction<idx_ij,idx_j>(A,vB);
    tester.test("A(i,j) * B(j)",D==E);
    D=A.transpose()*vB;
    E=impl.contraction<idx_ji,idx_j>(A,vB);
    tester.test("A(j,i) * B(j)",D==E);
    sD=sA*sB;
    sE=simpl.contraction<idx_i,idx_i>(sA,sB);
    tester.test("Scalar A * B", sD==sE);

    //Self-adjoint Eigen solver
    eigen_matrix L(2,2);
    L<<1, 2, 2, 3;
    Eigen::SelfAdjointEigenSolver<eigen_matrix> solver(L);
    auto eigen_sys=impl.self_adjoint_eigen_solver(L);

    tester.test("Eigenvalues",eigen_sys.first==solver.eigenvalues());
    tester.test("Eigenvectors",eigen_sys.second==solver.eigenvectors());



//    //Allocator
//    matrix_type allocated(shape);
//    tester.test("Allocate dimensions",allocated.shape()==corr_shape);

//    //Distribution
//    eigen_matrix P=E*(G+G.transpose());
//    matrix_type _P=_E(i,k)*(_G(k,j)+_G(j,i));
//    tester.test("E*(G+G^T)",_P==P);
    return tester.results();
}
