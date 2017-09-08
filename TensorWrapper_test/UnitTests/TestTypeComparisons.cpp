#include<TensorWrapper/TMUtils/TypeComparisons.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace detail_;

int main()
{
    Tester tester("Testing type comparisons");

    static_assert(TypeCount<int,int,int>::value==2,"TypeCount same types");
    static_assert(TypeCount<int,int,bool>::value==1,"TypeCount 1st type");
    static_assert(TypeCount<int,bool,int>::value==1,"TypeCount 2nd type");
    static_assert(TypeCount<bool,int,int>::value==0,"TypeCount not in list");

    using AllUnique1=typename GetUnique<int,double,bool>::type;
    using Unique1=typename GetUnique<int,bool,bool>::type;
    using Unique2=typename GetUnique<bool,int,bool>::type;
    using Unique3=typename GetUnique<bool,bool,int>::type;
    using NoUnique=typename GetUnique<int,int,int>::type;
    using Empty=std::tuple<>;
    using Tint=std::tuple<int>;

    static_assert(std::is_same<AllUnique1,std::tuple<int,double,bool>>::value,
                  "All unique");
    static_assert(std::is_same<Unique1,Tint>::value,"1st unique");
    static_assert(std::is_same<Unique2,Tint>::value,"2n unique");
    static_assert(std::is_same<Unique3,Tint>::value,"3rd unique");
    static_assert(std::is_same<NoUnique,Empty>::value,"No unique");

    using NoCommon=typename GetCommon<int,double,bool>::type;
    using Common23=typename GetCommon<int,bool,bool>::type;
    using Common13=typename GetCommon<bool,int,bool>::type;
    using Common12=typename GetCommon<bool,bool,int>::type;
    using AllCommon=typename GetCommon<int,int,int>::type;
    using Tbb=std::tuple<bool,bool>;

    static_assert(std::is_same<NoCommon,Empty>::value, "No common");
    static_assert(std::is_same<Common23,Tbb>::value,"23 common");
    static_assert(std::is_same<Common13,Tbb>::value,"13 cmmon");
    static_assert(std::is_same<Common12,Tbb>::value,"12 common");
    static_assert(std::is_same<AllCommon,std::tuple<int,int,int>>::value, "All common");

    static_assert(!AllUnique<int,int>::value,"All unique same");
    static_assert(AllUnique<>::value,"All unique empty");
    static_assert(AllUnique<int>::value,"All unique single");
    static_assert(AllUnique<int,bool>::value,"All unique different");

    return tester.results();
}
