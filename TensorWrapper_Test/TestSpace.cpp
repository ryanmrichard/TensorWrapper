#include <TensorWrapper/Space.hpp>
#include <UtilitiesEx/IterTools/Enumerate.hpp>
#include <catch/catch.hpp>
#include <numeric>


using namespace TensorWrapper;
using namespace UtilitiesEx;

template<std::size_t order>
using array_type=std::array<std::size_t, order>;

template<typename array_t>
void check_state(const Space& s, const array_t& lengths){
    const std::size_t size = std::accumulate(lengths.begin(), lengths.end(),
                                             1L, std::multiplies<>{});
    const std::size_t order = lengths.size();
    REQUIRE(s.size() == size);
    REQUIRE(s.order() == order);
    for(auto o : UtilitiesEx::Enumerate(lengths))
        REQUIRE(s.length(std::get<0>(o)) == std::get<1>(o));
    if(order)
        REQUIRE(!s.count(lengths));
}

TEST_CASE("Default Constructor"){
    Space s;
    check_state(s, array_type<0>{});
}


TEST_CASE("Length Construction"){
    std::vector<std::string> names{"scalar", "vector", "matrix", "tensor"};

    /*
     * Highs will be (3, 5, 7 ...)
     */

    for(std::size_t i=0; i<4; ++i){
        std::vector<std::size_t> highs(i);
        for(std::size_t j=0; j<i; ++j) highs[j]= 3 + 2*j;
        SECTION(names[i] + " space"){
                Space s{highs};
                check_state(s, highs);
        }
    }
}

TEST_CASE("Copy/Move construction/assignment"){
    array_type<2> lengths{3,4};
    Space s{lengths};
    SECTION("Copy construction"){
        Space copy(s);
        check_state(copy, lengths);
    }
    SECTION("Copy assignment"){
        Space empty;
        empty=s;
        check_state(empty, lengths);
    }
    SECTION("Move construction"){
        Space moved(std::move(s));
        check_state(moved, lengths);
    }
    SECTION("Move assignment"){
        Space empty;
        empty = std::move(s);
        check_state(empty, lengths);
    }
}

TEST_CASE("Shuffle") {
    array_type<3> lengths{3, 4, 9};
    array_type<0> null{};
    Space s{lengths};
    SECTION("Null shuffle"){
        s.shuffle(null, null);
        for(auto i : Range(3))
            REQUIRE(s.length(i) == lengths[i]);
    }
    array_type<3> to{0, 1, 2};
    array_type<3> from{1, 2, 0};
    SECTION("1->0 2->1 0->2"){
        s.shuffle(from, to);
        for(auto i: Range(3))
        REQUIRE(s.length(to[i])  == lengths[from[i]]);
    }
}

//TEST_CASE("Relations") {
//
//    std::array<std::size_t, 2> lows{0, 0};
//    std::array<std::size_t, 2> highs{3,3};
//
//    Space s{lows, highs};
//
//    SECTION("Self relations"){
//        REQUIRE(s==s);
//        REQUIRE(s<=s);
//        REQUIRE(!(s<s));
//        REQUIRE(!(s>s));
//        REQUIRE(s>=s);
//    }
//
//    SECTION("Empty relations"){
//        Space empty;
//        REQUIRE(empty != s);
//        REQUIRE(empty < s);
//        REQUIRE(empty <= s);
//        REQUIRE(!(empty > s));
//        REQUIRE(!(empty >= s));
//    }
//
//    SECTION("Superspace") {
//        highs[0]+=1;
//        Space su{lows, highs};
//        REQUIRE(s != su);
//        REQUIRE(s <= su);
//        REQUIRE(s <  su);
//        REQUIRE(!(s> su));
//        REQUIRE(!(s >= su));
//    }
//
//    SECTION("Subspace") {
//        highs[0]-=1;
//        Space su{lows, highs};
//        REQUIRE(s != su);
//        REQUIRE(!(s <= su));
//        REQUIRE(!(s <  su));
//        REQUIRE(s> su);
//        REQUIRE(s >= su);
//    }
//
//    SECTION("Lower Order Subspace") {
//        Space su{array_type<1>{0}, array_type<1>{3}};
//        REQUIRE(s != su);
//        REQUIRE(!(s <= su));
//        REQUIRE(!(s <  su));
//        REQUIRE(s> su);
//        REQUIRE(s >= su);
//    }
//
//
//}
