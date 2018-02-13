#include <TensorWrapper/Space.hpp>
#include <catch/catch.hpp>

using namespace TensorWrapper;

template<std::size_t order>
using array_type=std::array<std::size_t, order>;

template<typename array_t>
void check_state(const Space& s, const array_t& lows, const array_t& highs){
    std::size_t size = 1;
    for(std::size_t o=0; o<lows.size(); ++o)
        size*=(highs[o] - lows[o]);
    REQUIRE(s.size() == size);
    REQUIRE(s.order() == lows.size());
    REQUIRE(s.nsubspaces() == 1);
    REQUIRE(!s.empty());
    array_t index(lows);
    while(std::lexicographical_compare(index.begin(), index.end(),
                                       highs.begin(), highs.end()))
    {
        REQUIRE(s.count(lows));
        for(size_t i=1;i<=lows.size();++i)
        {
            const std::size_t idx = lows.size() - i;
            if(index[idx] < highs[idx])
            {
                ++index[idx];
                for(size_t j=1; j<i; ++j)
                    index[idx+j] = lows[idx+j];
            }

        }
    }
    if(lows.size())
    {
        REQUIRE(!s.count(highs));
        REQUIRE(!s.count(array_t(lows.size(), 99)));
    }
    else{
        REQUIRE(s.count(lows));
    }
    index.push_back(99);
    REQUIRE(!s.count(index));
}

TEST_CASE("Empty")
{
    Space s;
    REQUIRE(s.size() == 0);
    REQUIRE(s.order() == 0);
    REQUIRE(s.nsubspaces() == 0);
    REQUIRE(s.empty());
}

TEST_CASE("Tensors"){
    std::vector<std::string> names{"scalar", "vector", "matrix", "tensor"};

    /*
     * Lows will be (0, 0, ...)
     * Highs will be (3, 5, 7 ...)
     */

    for(std::size_t i=0; i<4; ++i){
        std::vector<std::size_t> lows(i);
        std::vector<std::size_t> highs(i);
        for(std::size_t j=0; j<i; ++j) highs[j]= 3 + 2*j;
        SECTION("Full space for: " + names[i]){
                Space s{highs};
                check_state(s, lows, highs);
        }
        for(std::size_t j=0; j<i; ++j)lows[j]=j+1;
        SECTION("Sub space for: " + names[i]){
            Space s{lows, highs};
            check_state(s, lows, highs);
        }
    }
}
