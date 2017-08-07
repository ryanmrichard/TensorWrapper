#include <TensorWrapper/TensorPtr.hpp>
#include "TestHelpers.hpp"
#include <vector>
using namespace TWrapper;

using tensor_ptr=detail_::TensorPtr<2,double>;
int main()
{
    Tester tester("Testing TensorPtr class");
    constexpr auto type=detail_::TensorTypes::EigenMatrix;
    Eigen::MatrixXd value=Eigen::MatrixXd::Zero(10,10);
    tensor_ptr defaulted;
    tester.test("Default constructor",!defaulted);

    tensor_ptr take_copy(type,value);
    tester.test("Copy Backend constructor",take_copy);
    tester.test("type",take_copy.type()==type);
    auto& wrapped_value=take_copy.cast<type>();
    tester.test("Take by value",wrapped_value==value);
    tester.test("Is a copy",&wrapped_value!=&value);

    tensor_ptr moved(type,std::move(value));
    auto& moved_value=moved.cast<type>();
    tester.test("RValue construct",moved_value==wrapped_value);

    tensor_ptr copied(moved);
    auto& copied_value=copied.cast<type>();
    tester.test("Copy constructed",copied_value==moved_value);
    tester.test("Is a copy",&copied_value!=&moved_value);

    tensor_ptr move_con(std::move(moved));
    auto &other_move=move_con.cast<type>();
    tester.test("Move constructor",&other_move==&moved_value);

    moved=move_con;
    auto& moved_value2=moved.cast<type>();
    tester.test("Copy assignment",moved_value2==other_move);
    tester.test("Is a copy",&moved_value2!=&other_move);

    moved=std::move(move_con);
    const auto& moved_value3=moved.cast<type>();
    tester.test("Move assignment",&moved_value3==&other_move);

    return tester.results();
}
