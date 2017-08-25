#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using matrix_type=EigenMatrix<double>;
using vector_type=EigenVector<double>;
using eigen_matrix=Eigen::MatrixXd;
using eigen_vector=Eigen::VectorXd;

int main()
{
    Tester tester("Testing Eigen Matrix Wrapping");
    const size_t dim=10;
    const std::array<size_t,2> shape({10,10});
    eigen_matrix A=eigen_matrix::Random(dim,dim),
                B=eigen_matrix::Random(dim,dim),
                C=eigen_matrix::Random(dim,dim);
    Shape<2> corr_shape(shape,false);

    //Allocator
    matrix_type allocated(shape);
    tester.test("Allocate dimensions",allocated.shape()==corr_shape);

    //Construction from Eigen Matrix
    matrix_type _A(A),_B(B),_C(C);
    tester.test("Values",_A==A);

    auto memory=_A.get_memory();
    tester.test("Memory shape",memory.local_shape==corr_shape);
    tester.test("Memory get",memory(3,3)==A(3,3));
    memory(3,3)=999.9;
    _A.set_memory(memory);
    tester.test("Memory set",_A(3,3)==999.9);
    A(3,3)=999.9;

    matrix_type slice=_A.slice({2,1},{3,3});
    tester.test("Slice shape",slice.shape()==Shape<2>({1,2},false));
    tester.test("Same element",slice(0,0)==A(2,1));

    //Basic operations
    eigen_matrix D=A+B+C;
    matrix_type _D=_A+_B+_C;
    tester.test("Addition",_D==D);

    eigen_matrix E=0.5*D;
    matrix_type _E=0.5*_D;
    tester.test("Left scale",_E==E);

    eigen_matrix F=D*0.5;
    matrix_type _F=_D*0.5;
    tester.test("Right scale",_F==F);

    eigen_matrix G=A-B-C;
    matrix_type _G=_A-_B-_C;
    tester.test("Subtraction",G==_G);


    //Matrix multiplications
    eigen_matrix H=G*E;
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");

    matrix_type _H=_G(i,j)*_E(j,k);
    tester.test("G * E",H==_H);

    eigen_matrix I=G.transpose()*E;
    matrix_type _I=_G(j,i)*_E(j,k);
    tester.test("G^T * E",I==_I);

    eigen_matrix J=G*E.transpose();
    matrix_type _J=_G(i,j)*_E(k,j);
    tester.test("G * E^T",J==_J);

    eigen_matrix K=G.transpose()*E.transpose();
    matrix_type _K=_G(j,i)*_E(k,j);
    tester.test("G^T * E^T",K==_K);

    eigen_matrix M=G.transpose()*E*G;
    matrix_type _M=_G(j,i)*_E(j,k)*_G(k,l);
    tester.test("G^T*E*G",M==_M);

    //Fancy basic ops
    eigen_matrix N=E+E.transpose();
    matrix_type _N=_E(i,j)+_E(j,i);
    tester.test("E+E^T",_N==N);

    eigen_matrix O=E-E.transpose();
    matrix_type _O=_E(i,j)-_E(j,i);
    tester.test("E-E^T",O==_O);

    //Distribution
    eigen_matrix P=E*(G+G.transpose());
    matrix_type _P=_E(i,k)*(_G(k,j)+_G(j,i));
    tester.test("E*(G+G^T)",_P==P);


    //Self-adjoint Eigen solver
    eigen_matrix L(2,2);
    L<<1, 2, 2, 3;
    Eigen::SelfAdjointEigenSolver<eigen_matrix> solver(L);
    matrix_type _L(L);
    auto eigen_sys=self_adjoint_eigen_solver(_L);

    tester.test("Eigenvalues",eigen_sys.first==solver.eigenvalues());
    tester.test("Eigenvectors",eigen_sys.second==solver.eigenvectors());


    const std::array<size_t,1> vshape({dim});
    eigen_vector vA=eigen_vector::Random(dim),
                 vB=eigen_vector::Random(dim),
                 vC=eigen_vector::Random(dim);
    Shape<1> vcorr_shape(vshape,false);

    //Basic Constructors
    vector_type vallocated(vshape);
    tester.test("Vector Allocate dimensions",vallocated.shape()==vcorr_shape);

    //Construction from Eigen Vector
    vector_type _vA(vA),_vB(vB),_vC(vC);
    tester.test("Vector Values",_vA==vA);

    auto vmemory=_vA.get_memory();
    tester.test("Vector Memory shape",vmemory.local_shape==vcorr_shape);
    tester.test("Vector Memory get",vmemory(3)==vA(3));
    vmemory(3)=999.9;
    _vA.set_memory(vmemory);
    tester.test("Vector Memory set",_vA(3)==999.9);
    vA(3)=999.9;
    vector_type vslice=_vA.slice({2},{3});
    tester.test("Vector Slice shape",vslice.shape()==Shape<1>({1},false));
    tester.test("Vector Same element",vslice(0)==vA(2));

    eigen_vector vD=vA+vB+vC;
    vector_type _vD=_vA+_vB+_vC;
    tester.test("Vector Addition",_vD==vD);

    eigen_vector vE=0.5*vD;
    vector_type _vE=0.5*_vD;
    tester.test("Vector Left scale",_vE==vE);

    eigen_vector vF=vD*0.5;
    vector_type _vF=_vD*0.5;
    tester.test("Vector Right scale",_vF==vF);

    eigen_vector vG=vA-vB-vC;
    vector_type _vG=_vA-_vB-_vC;
    tester.test("Vector Subtraction",vG==_vG);

    eigen_matrix vH=vG.transpose()*vG;
    EigenScalar<double> _vH=_vG(i)*_vG(i);
    tester.test("Vector dot",vH==_vH);

    eigen_matrix vI=vG*vG.transpose();
    matrix_type _vI=_vG(i)*_vG(j);
    tester.test("Vector outer product",_vI==vI);

    eigen_vector vJ=vG.transpose()*G;
    vector_type _vJ=_vG(i)*_G(i,j);
    tester.test("Vector times matrix",vJ==_vJ);

    eigen_vector vK=vG.transpose()*G.transpose();
    vector_type  _vK=_vG(i)*_G(j,i);
    tester.test("Vector times matrix^T",_vK==vK);

    eigen_vector vL=G*vG;
    vector_type _vL=_G(i,j)*_vG(j);
    tester.test("Matrix times vector",vL==_vL);

    eigen_vector vM=G.transpose()*vG;
    vector_type _vM=_G(i,j)*_vG(i);
    tester.test("Matrix^T times vector",vM==_vM);

    return tester.results();
}
