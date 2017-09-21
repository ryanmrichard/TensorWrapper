#include <TensorWrapper/IndexItr.hpp>
#include <vector>
#include "TestHelpers.hpp"

template<size_t rank>
using index_t=std::array<size_t,rank>;

template<size_t rank>
using corr_type=std::array<std::vector<index_t<rank>>,2>;

const corr_type<0> corr_D0{{
    {},//col major
    {}//row major
}};

const corr_type<1> corr_D1{{
    {{0},{1},{2},{3},{4}},//col major
    {{0},{1},{2},{3},{4}}//row major
}};

const corr_type<1> corr_D1_off{{
        {{1},{2},{3},{4}},//col major
        {{1},{2},{3},{4}}//row major
}};

const corr_type<2> corr_D2{{
        {{0,0},{1,0},{2,0},{3,0},{4,0},
         {0,1},{1,1},{2,1},{3,1},{4,1},
         {0,2},{1,2},{2,2},{3,2},{4,2},
         {0,3},{1,3},{2,3},{3,3},{4,3},
         {0,4},{1,4},{2,4},{3,4},{4,4}},//col major
        {{0,0},{0,1},{0,2},{0,3},{0,4},
         {1,0},{1,1},{1,2},{1,3},{1,4},
         {2,0},{2,1},{2,2},{2,3},{2,4},
         {3,0},{3,1},{3,2},{3,3},{3,4},
         {4,0},{4,1},{4,2},{4,3},{4,4}
        }//row major
}};

const corr_type<2> corr_D2_off{{
        {{0,1},{1,1},{2,1},{3,1},{4,1},
         {0,2},{1,2},{2,2},{3,2},{4,2},
         {0,3},{1,3},{2,3},{3,3},{4,3},
         {0,4},{1,4},{2,4},{3,4},{4,4}},//col major
        {{0,1},{0,2},{0,3},{0,4},
         {1,1},{1,2},{1,3},{1,4},
         {2,1},{2,2},{2,3},{2,4},
         {3,1},{3,2},{3,3},{3,4},
         {4,1},{4,2},{4,3},{4,4}
        }//row major
}};


using namespace TWrapper;

int main()
{
    Tester tester("Testing IndexItr class");

    //Rank 0 iterator
    const index_t<0> Zero0{};
    for(bool row_major: {true,false})
    {
        IndexItr<0> Itr(Zero0,true,row_major),
                    Itrend(Zero0,false,row_major);
        std::string prefix=(row_major?"Row-major":"Col-major");
        tester.test(prefix+" scalar itr start",*Itr==Zero0);
        tester.test(prefix+" scalar itr end",*Itrend==Zero0);
        std::vector<index_t<0>> vals;
        for(;Itr!=Itrend;++Itr)
           vals.push_back(*Itr);
        tester.test(prefix+" values",vals==corr_D0[row_major]);
    }


    //Rank 1 iterator
    const index_t<1> vec_size{5};
    const index_t<1> One1{1};
    const index_t<1> Zero1{};
    for(bool row_major: {true,false})
    {
        IndexItr<1> Itr(vec_size,true,row_major),
                    Itrend(vec_size,false,row_major);
        std::string prefix=(row_major?"Row-major":"Col-major");
        tester.test(prefix+" vector itr start",*Itr==Zero1);
        tester.test(prefix+" vector itr end",*Itrend==vec_size);
        std::vector<index_t<1>> vals;
        for(;Itr!=Itrend;++Itr)
            vals.push_back(*Itr);
        tester.test(prefix+" values",vals==corr_D1[row_major]);

        //With offset
        IndexItr<1> Itr2(vec_size,true,row_major,One1),
                    Itr2end(vec_size,false,row_major,One1);
        tester.test(prefix+" offset vector itr start",*Itr2==One1);
        tester.test(prefix+" offset vector itr end",*Itr2end==vec_size);
        std::vector<index_t<1>> vals2;
        for(;Itr2!=Itr2end;++Itr2)
            vals2.push_back(*Itr2);
        tester.test(prefix+" offset values",vals2==corr_D1_off[row_major]);
    }

    //Rank 2 iterator
    const index_t<2> mat_size{5,5};
    const index_t<2> One2{0,1};
    const index_t<2> Zero2{};
    for(bool row_major: {true,false})
    {
        IndexItr<2> Itr(mat_size,true,row_major),
                    Itrend(mat_size,false,row_major);
        std::string prefix=(row_major?"Row-major":"Col-major");
        tester.test(prefix+" matrix itr start",*Itr==Zero2);
        tester.test(prefix+" matrix itr end",*Itrend==mat_size);
        std::vector<index_t<2>> vals;
        for(;Itr!=Itrend;++Itr)
            vals.push_back(*Itr);
        tester.test(prefix+" matrix values",vals==corr_D2[row_major]);

        //With offset
        IndexItr<2> Itr2(mat_size,true,row_major,One2),
                    Itr2end(mat_size,false,row_major,One2);
        tester.test(prefix+" offset matrix itr start",*Itr2==One2);
        tester.test(prefix+" offset matrix itr end",*Itr2end==mat_size);
        std::vector<index_t<2>> vals2;
        for(;Itr2!=Itr2end;++Itr2)
            vals2.push_back(*Itr2);
        tester.test(prefix+" offset matrix values",
                    vals2==corr_D2_off[row_major]);
    }


    return tester.results();
}
