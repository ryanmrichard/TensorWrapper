#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"


/** \file This file tests my meta-template programming infrfdastructure.  Unlike
 *  the other tests you don't really need to run this one to make sure it works.
 *  The reason is if it doesn't work this test won't compile.
 *
 */

using namespace TWrapper;
using namespace detail_;

int main()
{
    Tester tester("Testing TensorWrapper traits infrastructure");
    constexpr auto type=TensorTypes::EigenMatrix;
    static_assert(!TensorTraits<double>::value,"False positive on is a tensor");
    using TW=TensorWrapper<2,double,type>;
    using TWB=TensorWrapperBase<2,double>;
    using TWTraits=TensorTraits<TW>;
    using TWBTraits=TensorTraits<TWB>;
    static_assert(TWTraits::value,"Failed to find tensor");
    static_assert(TWTraits::rank==2,"Wrong rank");
    static_assert(
        std::is_convertible<typename TWTraits::scalar_type,double>::value,
                "Wrong scalar type");
    static_assert(TWBTraits::value,"Failed to find tensor");
    static_assert(TWBTraits::rank==2,"Wrong rank");
    static_assert(
        std::is_convertible<typename TWBTraits::scalar_type,double>::value,
                "Wrong scalar type");
    static_assert(std::is_convertible<CleanType<double>,double>::value,
                  "Failed to recognize an already claen type");
    static_assert(std::is_convertible<CleanType<const double>,double>::value,
                  "Failed to remove const");
    static_assert(std::is_convertible<CleanType<double&>,double>::value,
                  "Failed to remove reference");
    static_assert(std::is_convertible<CleanType<const double&>,double>::value,
                  "Failed to remove const and/or reference");
    static_assert(!IsATWrapper<double>::value,"False positive on is a TW");
    static_assert(IsATWrapper<const TW>::value,"Failed to recognize a TW");
    static_assert(!OperationTraits<double>::value,"False positive on op");

    using Op=OperationBase<double>;
    using DerivedOp=Convert<double>;
    static_assert(OperationTraits<Op>::value,"Failed to recognize an op");
    static_assert(OperationTraits<DerivedOp>::value,"Failed to recognize op");
    static_assert(!IsAnOperation<double>::value,"False positive on op");
    static_assert(IsAnOperation<const Op>::value,"Failed to recognize an op");
    static_assert(std::is_convertible<
                  typename EnableIfAnOperation<const Op>::type,int>::value,
                  "False positive on operation");
    static_assert(std::is_convertible<
                  typename EnableIfNotAnOperation<double>::type,int>::value,
                  "False positive on operation");


    using NotAnOpOrTW=IsOpOrTW<double>;
    static_assert(!NotAnOpOrTW::is_op,"False positive on op");
    static_assert(!NotAnOpOrTW::is_tw,"False positive on TW");
    static_assert(!NotAnOpOrTW::value,"Incorrectly combined booleans");
    using IsOp=IsOpOrTW<Op>;
    using IsOp2=IsOpOrTW<DerivedOp>;
    using IsTW=IsOpOrTW<TW>;
    static_assert(IsOp::is_op,"Failed to recognize an op");
    static_assert(!IsOp::is_tw,"False positive on TW");
    static_assert(IsOp::value,"Incorrectly combined booleans");
    static_assert(IsOp2::is_op,"Failed to recognize an op");
    static_assert(!IsOp2::is_tw,"False positive on TW");
    static_assert(IsOp2::value,"Incorrectly combined booleans");
    static_assert(!IsTW::is_op,"False positive on an op");
    static_assert(IsTW::is_tw,"Failed to recognize a TW");
    static_assert(IsTW::value,"Incorrectly combined booleans");
    static_assert(std::is_convertible<
                  typename EnableIfNotOpOrTW<const double>::type,int>::value,
                  "False positive on Op or TW");
    static_assert(std::is_convertible<
                  typename EnableIfOpOrTW<Op>::type,int>::value,
                  "Failed to find an operation");
    static_assert(std::is_convertible<
                  typename EnableIfOpOrTW<TW>::type,int>::value,
                  "Failed to find a TW");
    return tester.results();
}
