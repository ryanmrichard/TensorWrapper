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
    Eigen::MatrixXd A=Eigen::MatrixXd::Random(10,10);

    EigenMatrix<double> _A;
    tester.test("Default rank",_A.rank()==2);

    EigenMatrix<double> B(A);
    auto& wrapped=B.data();
    tester.test("Construct from native instance",wrapped==A);

    TensorWrapperBase<2,double>& base=B;
    auto pclone=base.clone();
    EigenMatrix<double>& clone=static_cast<EigenMatrix<double>&>(*pclone);
    tester.test("Clone through base reference",clone.data()==wrapped);
    tester.test("Clone is copy",&(clone.data())!=&wrapped);

    ///TODO: Test conversion constructor

    EigenMatrix<double> copy_of_B(B);
    tester.test("Copy constructor",copy_of_B.data()==wrapped);
    tester.test("Is a copy",&(copy_of_B.data())!=&wrapped);

    EigenMatrix<double> movedB(std::move(B));
    tester.test("Move constructor",&(movedB.data())==&wrapped);

    B=movedB;
    tester.test("Assignment",B.data()==wrapped);
    tester.test("Assignment copies",&(B.data())!=&wrapped);

    B=std::move(movedB);
    tester.test("Move assignment",&(B.data())==&wrapped);

    return tester.results();
}
