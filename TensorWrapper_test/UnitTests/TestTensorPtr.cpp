#include <TensorWrapper/TensorPtr.hpp>
#include "TestHelpers.hpp"
#include <vector>
using namespace TWrapper;

using tensor_ptr=detail_::TensorPtr<2,double>;
/* Note that although we call it TensorPtr, it really can hold anything, we
 * abuse that fact in this test.
 */
int main()
{
    Tester tester("Testing TensorPtr class");
    constexpr auto type=detail_::TensorTypes::EigenMatrix;
    Eigen::MatrixXd value=Eigen::MatrixXd::Zero(10,10);
    tensor_ptr defaulted;
    tester.test("Implicit default",!defaulted);
    tensor_ptr take_copy(type,value);
    tester.test("Implicit copy",take_copy);
    auto& wrapped_value=take_copy.cast<type>();
    tester.test("Take by value",wrapped_value==value);
    auto temp=wrapped_value+wrapped_value;
    tester.test("Return is usable",temp==value);
    tensor_ptr moved(type,std::move(value));
    auto& moved_value=moved.cast<type>();
    tester.test("Move",moved_value==wrapped_value);
    tensor_ptr copied(moved);
    auto& copied_value=copied.cast<type>();
    tester.test("Copy",copied_value==moved_value);
    return tester.results();
}
