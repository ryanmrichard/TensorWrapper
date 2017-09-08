#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

/** \file This file contains tests for the TensorWrapper and TensorWrapperBase
 *  classes.  We test them together because one cannot make a usable
 *  TensorWrapperBase instance without going through the TensorWrapper API.
 *
 *  These tests use the Eigen backend, but given that TensorWrapper goes through
 *  the TensorWrapperImpl API regardless of backend (and that each of those
 *  wrappings is tested elsewhere) this test passing should really reflect that
 *  our API is working regardless of backend.
 *
 */


using namespace TWrapper;

int main()
{
    Tester tester("Testing TensorWrapper and TensorWrapperBase classes");
    using index_t=typename EigenMatrix<double>::index_t;

    EigenMatrix<double> A;
    tester.test("Default rank",A.rank()==2);

    Eigen::MatrixXd _A=Eigen::MatrixXd::Random(10,10);
    EigenMatrix<double> B(_A);
    auto& wrapped=B.data();
    tester.test("Construct from native instance",wrapped==_A);

    A=B;
    tester.test("Assignment",B.data()==wrapped);
    tester.test("Assignment copies",&wrapped!=&(A.data()));

    TensorWrapperBase<2,double>& base=B;
    auto pclone=base.clone();

    //Down cast because we have a TW base instance and we want to test TW API
    EigenMatrix<double>& clone=static_cast<EigenMatrix<double>&>(*pclone);
    tester.test("Clone through base reference",clone.data()==wrapped);
    tester.test("Clone is copy",&(clone.data())!=&wrapped);

    tester.test("Equality comparison",A==B);
    tester.test("Equality b/w TW and native",B==_A);
    tester.test("Equality b/w native and TW",_A==B);
    tester.test("Inequality",!(A!=B));
    tester.test("Inequality b/w TW and native",!(B!=_A));
    tester.test("Inequality b/w native and TW",!(_A!=B));

    EigenMatrix<double> copy_of_B(B);
    tester.test("Copy constructor",copy_of_B.data()==wrapped);
    tester.test("Is a copy",&(copy_of_B.data())!=&wrapped);

    EigenMatrix<double> movedB(std::move(B));
    tester.test("Move constructor",&(movedB.data())==&wrapped);

    B=std::move(movedB);
    tester.test("Move assignment",&(B.data())==&wrapped);

    EigenTensor<2,double> C(index_t{10,10});
    fill_random(C);
    tester.test("Allocate constructor",&C.data()!=nullptr);

    EigenMatrix<double> D(C);
    {
        auto& d=D.data();
        auto& c=C.data();
        tester.test("Converted",std::equal(d.data(),d.data()+100,c.data()));
    }

    C=D;
    {
        auto& d=D.data();
        auto& c=C.data();
        tester.test("Convt assign",std::equal(d.data(),d.data()+100,c.data()));
    }

    Shape<2> corr_shape(index_t{10,10},false);
    tester.test("Get shape",A.shape()==corr_shape);

    const double a=_A(0,0);
    tester.test("Array element access",A(index_t{0,0})==a);
    tester.test("Variadic element access",A(0,0)==a);

    EigenMatrix<double> J=A.slice(index_t{0,0},index_t{5,5});
    Eigen::MatrixXd K=_A.block(0,0,5,5);
    tester.test("Slice",J==K);

    auto Amem=A.get_memory();
    tester.test("Get memory",Amem.block(0)==A.data().data());
    Amem.block(0)[0]=2.1;
    A.set_memory(Amem);
    tester.test("Set memory",A(0,0)==2.1);

    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");

    //This implicitly tests our construct from Operation constructor
    EigenMatrix<double> E=A(i,j)+B(i,j);
    Eigen::MatrixXd F=A.data()+B.data();
    tester.test("Addition w/ indices",E==F);


    //These implicity test our ability to assign from Operation
    E=A+B;
    tester.test("Addition w/o indices",E==F);
    E=A(i,j)+B(j,i);
    F=A.data()+B.data().transpose();
    tester.test("Addition w/ transpose",E==F);

    E=A(i,j)-B(i,j);
    F=A.data()-B.data();
    tester.test("Subtraction w/ indices",E==F);
    E=A-B;
    tester.test("Subtraction w/o indices",E==F);
    E=A(i,j)-B(j,i);
    F=A.data()-B.data().transpose();
    tester.test("Subtraction w/ transpose",E==F);

    E=0.5*A(i,j);
    F=0.5*A.data();
    tester.test("Left scale w/indices",E==F);
    E=0.5*A;
    tester.test("Left scale w/o indices",E==F);
    E=A(i,j)*0.5;
    tester.test("Right scale w/ indices",E==F);
    E=A*0.5;
    tester.test("Right scale w/o indices",E==F);

    E=A(i,j)*B(j,k);
    F=A.data()*B.data();
    tester.test("Contraction",E==F);

    EigenScalar<double> G=A(i,i);
    double h=A.data().trace();
    tester.test("Trace",G(std::array<size_t,0>{})==h);





    return tester.results();
}
