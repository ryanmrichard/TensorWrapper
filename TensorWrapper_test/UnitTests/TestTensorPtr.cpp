#include <TensorWrapper/TensorPtr.hpp>
#include "TestHelpers.hpp"
#include <vector>
using namespace TWrapper;


/* Note that although we call it TensorPtr, it really can hold anything, we
 * abuse that fact in this test.
 */
int main()
{
    Tester tester("Testing TensorPtr class");

    detail_::TensorPtr defaulted;
    tester.test("Implicit default",!defaulted);
    std::vector<double> value({5.8});
    detail_::TensorPtr take_double(value);
    tester.test("Implicit conversion",take_double);
    auto& wrapped_value=take_double.cast<std::vector<double>>();
    tester.test("Take by value",wrapped_value==value);
    detail_::TensorPtr moved(std::move(value));
    auto& moved_value=moved.cast<std::vector<double>>();
    tester.test("Move",moved_value==wrapped_value);
    detail_::TensorPtr copied(moved);
    auto& copied_value=copied.cast<std::vector<double>>();
    tester.test("Copy",copied_value==moved_value);
    return tester.results();
}
