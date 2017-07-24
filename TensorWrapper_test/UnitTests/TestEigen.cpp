#include <TensorWrapper/TensorWrapper.hpp>
#include <TensorWrapper/Operation.hpp>
#include<TensorWrapper/TensorImpl/TensorTypes.hpp>
#include <iostream>
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

    //Basic Constructors
    matrix_type defaulted;
    tester.test("Default rank",defaulted.rank()==2);
    matrix_type allocated(shape);
    tester.test("Allocate rank",allocated.rank()==2);
    tester.test("Allocate dimensions",allocated.shape()==corr_shape);
    matrix_type moved(std::move(allocated));
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
    matrix_type _A(A),_B(B),_C(C);
    tester.test("Rank",_A.rank()==2);
    tester.test("Dimensions",_A.shape()==corr_shape);
    tester.test("Values",_A==A);

    eigen_matrix D=A+B+C;
    matrix_type _D=_A+_B;//+_C;
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


    auto memory=_D.get_memory();
    tester.test("Memory shape",memory.local_shape==corr_shape);
    tester.test("Memory get",memory(3,3)==D(3,3));
    memory(3,3)=999.9;
    _D.set_memory(memory);
    tester.test("Memory set",_D(3,3)==999.9);

    matrix_type slice=_D.slice({2,1},{3,3});
    tester.test("Slice shape",slice.shape()==Shape<2>({1,2},false));
    tester.test("Same element",slice(0,0)==D(2,1));

//    eigen_matrix H=G*E;
//    matrix_type<2> _H=_G("i,j")*_E("j,k");
//    tester.test("G * E",H==_H);

//    eigen_matrix I=G.transpose()*E;
//    matrix_type<2> _I=_G("j,i")*_E("j,k");
//    tester.test("G^T * E",I==_I);

//    eigen_matrix J=G*E.transpose();
//    matrix_type<2> _J=_G("i,j")*_E("k,j");
//    tester.test("G * E^T",J==_J);

//    eigen_matrix K=G.transpose()*E.transpose();
//    matrix_type<2> _K=_G("j,i")*_E("k,j");
//    tester.test("G^T * E^T",K==_K);

//    //Self-adjoint Eigen solver
//    eigen_matrix L(2,2);
//    L<<1, 2, 2, 3;
//    Eigen::SelfAdjointEigenSolver<eigen_matrix> solver(L);
//    matrix_type<2> _L(L);
//    auto eigen_sys=self_adjoint_eigen_solver(_L);
//    const Shape<1> vec_shape({2},false);

//    tester.test("Vector dims",eigen_sys.first.dims()==vec_shape);
//    tester.test("Eigenvalues",eigen_sys.first==solver.eigenvalues());
//    tester.test("Eigenvectors",eigen_sys.second==solver.eigenvectors());


    const std::array<size_t,1> vshape({dim});
    eigen_vector vA=eigen_vector::Random(dim),
                 vB=eigen_vector::Random(dim),
                 vC=eigen_vector::Random(dim);
    Shape<1> vcorr_shape(vshape,false);

    //Basic Constructors
    vector_type vdefaulted;
    tester.test("Vector default rank",vdefaulted.rank()==1);
    vector_type vallocated(vshape);
    tester.test("Vector Allocate rank",vallocated.rank()==1);
    tester.test("Vector Allocate dimensions",vallocated.shape()==vcorr_shape);
    vector_type vmoved(std::move(vallocated));
    tester.test("Vector Moved rank",vmoved.rank()==1);
    tester.test("Vector Moved dimensions",vmoved.shape()==vcorr_shape);

    //Basic Assignment
    vallocated=vmoved;
    tester.test("Vector Assignment rank",vallocated.rank()==1);
    tester.test("Vector Assignment dimensions",vallocated.shape()==vcorr_shape);
    vdefaulted=std::move(vmoved);
    tester.test("Vector Move allocate rank",vdefaulted.rank()==1);
    tester.test("Vector Allocate dimensions",vdefaulted.shape()==vcorr_shape);


    //Construction from Eigen Vector
    vector_type _vA(vA),_vB(vB),_vC(vC);
    tester.test("Vector Rank",_vA.rank()==1);
    tester.test("Vector Dimensions",_vA.shape()==vcorr_shape);
    tester.test("Vector Values",_vA==vA);

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

    auto vmemory=_vD.get_memory();
    tester.test("Vector Memory shape",vmemory.local_shape==vcorr_shape);
    tester.test("Vector Memory get",vmemory(3)==vD(3));
    vmemory(3)=999.9;
    _vD.set_memory(vmemory);
    tester.test("Vector Memory set",_vD(3)==999.9);

    vector_type vslice=_vD.slice({2},{3});
    tester.test("Vector Slice shape",vslice.shape()==Shape<1>({1},false));
    tester.test("Vector Same element",vslice(0)==vD(2));


    return tester.results();
}
