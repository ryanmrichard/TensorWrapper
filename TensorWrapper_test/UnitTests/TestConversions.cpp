#include <TensorWrapper/TensorWrapper.hpp>
#include <iostream>
#include "TestHelpers.hpp"

//using namespace TWrapper;
//using eigen_matrix=TensorWrapper<2,double,detail_::TensorTypes::EigenMatrix>;
//using eigen_tensor=TensorWrapper<2,double,detail_::TensorTypes::EigenTensor>;
//using wrapped_type=eigen_matrix::wrapped_t;

int main()
{
    Tester tester("Testing Conversions");
//    const size_t dim=10;
//    const std::array<size_t,2> dims({dim,dim});
//    wrapped_type A=wrapped_type::Random(dim,dim);
//    eigen_matrix B(A);
//    eigen_tensor C(B);
//    tester.test("Eigen matrix to eigen tensor",C==B);
//    eigen_matrix D(C);
//    tester.test("Eigen tensor to eigen matrix",D==C);

    return tester.results();
}
