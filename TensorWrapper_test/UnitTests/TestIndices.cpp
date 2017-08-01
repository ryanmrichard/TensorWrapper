#include <TensorWrapper/Indices.hpp>
#include "TestHelpers.hpp"

/** \file This file tests our compile time string parsing.  Similar to the
 *  TestTraits test one does not actually have to run this test as the fact
 *  that it compiles means it works.
 */


using namespace TWrapper::detail_;

//Function to make sure we have the API I want
template<typename...Args>
Indices<Args...> test_fxn(Args...)
{
    return Indices<Args...>{};
}

//Comparing arrays is not a constexpr, this function makes it one
template<typename T, size_t R>
constexpr bool are_equal(const std::array<T,R>& lhs,
                         const std::array<T,R>& rhs)
{
    for(size_t i=0;i<R;++i)
        if(lhs[i]!=rhs[i])
            return false;
    return true;
}


int main()
{

    Tester tester("Testing Indices");

     //C_String class instances to be used throughout
     auto i=make_index("i");
     auto j=make_index("j");
     auto k=make_index("k");
     auto l=make_index("l");

     //Indices class instances to be used throughout test
     //Note: 1 and 3 can't actually be contracted against eachother, as i
     //appears 3 times in the resulting term, nevertheless must of the functions
     //are general enough that this won't break them so we still check this term
     //pairing anyways.
     auto Idx1 = test_fxn(i,j,k);
     auto Idx2 = test_fxn(j,l,k);
     auto Idx3 = test_fxn(i,i,k,j); //Used to test common index in single tensor



     //Size function
     static_assert(Idx1.size()==3," Idx1 size");
     static_assert(Idx2.size()==3," Idx2 size");
     static_assert(Idx3.size()==4," Idx3 size");

     //Get index
     static_assert(Idx1.get<0>()==i,"Idx1 0");
     static_assert(Idx1.get<1>()==j,"Idx1 1");
     static_assert(Idx1.get<2>()==k,"Idx1 2");
     static_assert(Idx2.get<0>()==j,"Idx2 0");
     static_assert(Idx2.get<1>()==l,"Idx2 1");
     static_assert(Idx2.get<2>()==k,"Idx2 2");
     static_assert(Idx3.get<0>()==i,"Idx3 0");
     static_assert(Idx3.get<1>()==i,"Idx3 1");
     static_assert(Idx3.get<2>()==k,"Idx3 2");
     static_assert(Idx3.get<3>()==j,"Idx3 3");

     //Position of index
     static_assert(Idx1.position(0,i)==0,"Idx1 position of i");
     static_assert(Idx1.position(0,j)==1,"Idx1 position of j");
     static_assert(Idx1.position(0,k)==2,"Idx1 position of k");
     static_assert(Idx2.position(0,j)==0,"Idx2 position of j");
     static_assert(Idx2.position(0,l)==1,"Idx2 position of l");
     static_assert(Idx2.position(0,k)==2,"Idx2 position of k");
     static_assert(Idx3.position(0,i)==0,"Idx3 position of first i");
     static_assert(Idx3.position(1,i)==1,"Idx3 position of second i");
     static_assert(Idx3.position(0,k)==2,"Idx3 position of k");
     static_assert(Idx3.position(0,j)==3,"Idx3 position of j");


     //Count function
     static_assert(Idx1.count(i)==1,"Idx1 count i");
     static_assert(Idx1.count(j)==1,"Idx1 count j");
     static_assert(Idx1.count(k)==1,"Idx1 count k");
     static_assert(Idx1.count(l)==0,"Idx1 count l (DNE)");
     static_assert(Idx2.count(i)==0,"Idx2 count i (DNE)");
     static_assert(Idx2.count(j)==1,"Idx2 count j");
     static_assert(Idx2.count(k)==1,"Idx2 count k");
     static_assert(Idx2.count(l)==1,"Idx2 count l");
     static_assert(Idx3.count(i)==2,"Idx3 count i");
     static_assert(Idx3.count(j)==1,"Idx3 count j");
     static_assert(Idx3.count(k)==1,"Idx3 count k");
     static_assert(Idx3.count(l)==0,"Idx3 count l (DNE)");

     //Get counts function
     constexpr std::array<size_t,3> count1{1,1,1};
     constexpr std::array<size_t,4> count3{2,2,1,1};
     static_assert(are_equal(Idx1.get_counts(),count1),"Idx1 get counts");
     static_assert(are_equal(Idx2.get_counts(),count1),"Idx2 get counts");
     static_assert(are_equal(Idx3.get_counts(),count3),"Idx3 get counts");

     //Number of common indices
     static_assert(Idx1.ncommon(Idx1)==3,"Idx1 ncommon Idx1");
     static_assert(Idx2.ncommon(Idx2)==3,"Idx2 ncommon Idx2");
     static_assert(Idx3.ncommon(Idx3)==4,"Idx3 ncommon Idx3");
     static_assert(Idx1.ncommon(Idx2)==2,"Idx1 ncommon Idx2");
     static_assert(Idx2.ncommon(Idx1)==2,"Idx2 ncommon Idx1");
     static_assert(Idx1.ncommon(Idx3)==3,"Idx1 ncommon Idx3");
     static_assert(Idx3.ncommon(Idx1)==4,"Idx3 ncommon Idx1");
     static_assert(Idx2.ncommon(Idx3)==2,"Idx2 ncommon Idx3");
     static_assert(Idx3.ncommon(Idx2)==2,"Idx3 ncommon Idx2");

     //Check getting common indices
     static_assert(Idx1.ith_common(0,Idx1)==0,"Idx1[0] common Idx1");
     static_assert(Idx1.ith_common(1,Idx1)==1,"Idx1[1] common Idx1");
     static_assert(Idx1.ith_common(2,Idx1)==2,"Idx1[2] common Idx1");
     static_assert(Idx2.ith_common(0,Idx2)==0,"Idx2[0] common Idx2");
     static_assert(Idx2.ith_common(1,Idx2)==1,"Idx2[1] common Idx2");
     static_assert(Idx2.ith_common(2,Idx2)==2,"Idx2[2] common Idx2");
     static_assert(Idx3.ith_common(0,Idx3)==0,"Idx3[0] common Idx3");
     static_assert(Idx3.ith_common(1,Idx3)==1,"Idx3[1] common Idx3");
     static_assert(Idx3.ith_common(2,Idx3)==2,"Idx3[2] common Idx3");
     static_assert(Idx3.ith_common(3,Idx3)==3,"Idx3[3] common Idx3");
     static_assert(Idx1.ith_common(0,Idx2)==1,"Idx1[0] common Idx2");
     static_assert(Idx1.ith_common(1,Idx2)==2,"Idx1[1] common Idx2");
     static_assert(Idx2.ith_common(0,Idx1)==0,"Idx2[0] common Idx1");
     static_assert(Idx2.ith_common(1,Idx1)==2,"Idx2[1] common Idx1");
     static_assert(Idx1.ith_common(0,Idx3)==0,"Idx1[0] common Idx3");
     static_assert(Idx1.ith_common(1,Idx3)==1,"Idx1[1] common Idx3");
     static_assert(Idx1.ith_common(2,Idx3)==2,"Idx1[2] common Idx3");
     static_assert(Idx3.ith_common(0,Idx1)==0,"Idx3[0] common Idx1");
     static_assert(Idx3.ith_common(1,Idx1)==1,"Idx3[1] common Idx1");
     static_assert(Idx3.ith_common(2,Idx1)==2,"Idx3[2] common Idx1");
     static_assert(Idx3.ith_common(3,Idx1)==3,"Idx3[3] common Idx1");
     static_assert(Idx2.ith_common(0,Idx3)==0,"Idx2[0] common Idx3");
     static_assert(Idx2.ith_common(1,Idx3)==2,"Idx2[1] common Idx3");
     static_assert(Idx3.ith_common(0,Idx2)==2,"Idx3[0] common Idx2");
     static_assert(Idx3.ith_common(1,Idx2)==3,"Idx2[1] common Idx3");

     //Check getting all common_indices
     constexpr std::array<size_t,3> com11{0,1,2};//also 22, 13
     constexpr std::array<size_t,4> com33{0,1,2,3};//also 31
     constexpr std::array<size_t,2> com12{1,2};
     constexpr std::array<size_t,2> com21{0,2};//also 23
     constexpr std::array<size_t,2> com32{2,3};
     static_assert(are_equal(com11,Idx1.get_common(Idx1)),"Idx1 common Idx1");
     static_assert(are_equal(com11,Idx2.get_common(Idx2)),"Idx2 common Idx2");
     static_assert(are_equal(com33,Idx3.get_common(Idx3)),"Idx3 common Idx3");
     static_assert(are_equal(com12,Idx1.get_common(Idx2)),"Idx1 common Idx2");
     static_assert(are_equal(com21,Idx2.get_common(Idx1)),"Idx2 common Idx1");
     static_assert(are_equal(com11,Idx1.get_common(Idx3)),"Idx1 common Idx3");
     static_assert(are_equal(com33,Idx3.get_common(Idx1)),"Idx3 common Idx1");
     static_assert(are_equal(com21,Idx2.get_common(Idx3)),"Idx2 common Idx3");
     static_assert(are_equal(com32,Idx3.get_common(Idx2)),"Idx3 common Idx2");

     //Number of unique indices
     static_assert(Idx1.nunique(Idx1)==0,"Idx1 nunique Idx1");
     static_assert(Idx2.nunique(Idx2)==0,"Idx2 nunique Idx2");
     static_assert(Idx3.nunique(Idx3)==0,"Idx3 nunique Idx3");
     static_assert(Idx1.nunique(Idx2)==1,"Idx1 nunique Idx2");
     static_assert(Idx2.nunique(Idx1)==1,"Idx2 nunique Idx1");
     static_assert(Idx1.nunique(Idx3)==0,"Idx1 nunique Idx3");
     static_assert(Idx3.nunique(Idx1)==0,"Idx3 nunique Idx1");
     static_assert(Idx2.nunique(Idx3)==1,"Idx2 nunique Idx3");
     static_assert(Idx3.nunique(Idx2)==2,"Idx3 nunique Idx2");

     //Get unique indices
     static_assert(Idx1.ith_unique(0,Idx2)==0,"Idx1[0] unique Idx2");
     static_assert(Idx2.ith_unique(0,Idx1)==1,"Idx2[0] unique Idx1");
     static_assert(Idx2.ith_unique(0,Idx3)==1,"Idx2[0] unique Idx3");
     static_assert(Idx3.ith_unique(0,Idx2)==0,"Idx3[0] unique Idx2");
     static_assert(Idx3.ith_unique(1,Idx2)==1,"Idx3[1] unique Idx2");

     //Check getting all unique indices
     constexpr std::array<size_t,0> unq11{};//also 22, 33, 13, and 31
     constexpr std::array<size_t,1> unq12{0};
     constexpr std::array<size_t,1> unq21{1};//also 23
     constexpr std::array<size_t,2> unq32{0,1};
     static_assert(are_equal(unq11,Idx1.get_unique(Idx1)),"Idx1 unique Idx1");
     static_assert(are_equal(unq11,Idx2.get_unique(Idx2)),"Idx2 unique Idx2");
     static_assert(are_equal(unq11,Idx3.get_unique(Idx3)),"Idx3 unique Idx3");
     static_assert(are_equal(unq12,Idx1.get_unique(Idx2)),"Idx1 unique Idx2");
     static_assert(are_equal(unq21,Idx2.get_unique(Idx1)),"Idx2 unique Idx1");
     static_assert(are_equal(unq11,Idx1.get_unique(Idx3)),"Idx1 unique Idx3");
     static_assert(are_equal(unq11,Idx3.get_unique(Idx1)),"Idx3 unique Idx1");
     static_assert(are_equal(unq21,Idx2.get_unique(Idx3)),"Idx2 unique Idx3");
     static_assert(are_equal(unq32,Idx3.get_unique(Idx2)),"Idx3 unique Idx2");

     constexpr auto free11=get_free(Idx1,Idx1);
     constexpr auto free22=get_free(Idx2,Idx2);
     constexpr auto free33=get_free(Idx3,Idx3);
     constexpr auto free12=get_free(Idx1,Idx2);
     constexpr auto free21=get_free(Idx2,Idx1);
     constexpr auto free13=get_free(Idx1,Idx3);
     constexpr auto free31=get_free(Idx3,Idx1);
     constexpr auto free23=get_free(Idx2,Idx3);
     constexpr auto free32=get_free(Idx3,Idx2);
     static_assert(are_equal(free11.first,unq11),"Idx1 free Idx1[0]");
     static_assert(are_equal(free11.second,unq11),"Idx1 free Idx1[1]");
     static_assert(are_equal(free22.first,unq11),"Idx2 free Idx2[0]");
     static_assert(are_equal(free22.second,unq11),"Idx2 free Idx2[1]");
     static_assert(are_equal(free33.first,unq11),"Idx3 free Idx3[0]");
     static_assert(are_equal(free33.second,unq11),"Idx3 free Idx3[1]");
     static_assert(are_equal(free12.first,unq12),"Idx1 free Idx2[0]");
     static_assert(are_equal(free12.second,unq21),"Idx1 free Idx2[1]");
     static_assert(are_equal(free21.first,unq21),"Idx2 free Idx1[0]");
     static_assert(are_equal(free21.second,unq12),"Idx2 free Idx1[1]");
     static_assert(are_equal(free13.first,unq11),"Idx1 free Idx3[0]");
     static_assert(are_equal(free13.second,unq11),"Idx1 free Idx3[1]");
     static_assert(are_equal(free31.first,unq11),"Idx3 free Idx1[0]");
     static_assert(are_equal(free31.second,unq11),"Idx3 free Idx1[1]");
     static_assert(are_equal(free23.first,unq21),"Idx2 free Idx3[0]");
     static_assert(are_equal(free23.second,unq32),"Idx2 free Idx3[1]");
     static_assert(are_equal(free32.first,unq32),"Idx3 free Idx2[0]");
     static_assert(are_equal(free32.second,unq21),"Idx3 free Idx2[1]");

     constexpr auto dummy11=get_dummy(Idx1,Idx1);
     constexpr auto dummy22=get_dummy(Idx2,Idx2);
     constexpr auto dummy12=get_dummy(Idx1,Idx2);
     constexpr auto dummy21=get_dummy(Idx2,Idx1);
     constexpr auto dummy23=get_dummy(Idx2,Idx3);
     constexpr auto dummy32=get_dummy(Idx3,Idx2);
     constexpr std::array<size_t,2> dum23{3,2};
     constexpr std::array<size_t,2> dum32{2,0};
     static_assert(are_equal(dummy11.first,com11),"Idx1 dummy Idx1[0]");
     static_assert(are_equal(dummy11.second,com11),"Idx1 dummy Idx1[1]");
     static_assert(are_equal(dummy22.first,com11),"Idx2 dummy Idx2[0]");
     static_assert(are_equal(dummy22.second,com11),"Idx2 dummy Idx2[1]");
     static_assert(are_equal(dummy12.first,com12), "Idx1 dummy Idx2[0]");
     static_assert(are_equal(dummy12.second,com21),"Idx1 dummy Idx2[1]");
     static_assert(are_equal(dummy21.first,com21),"Idx2 dummy Idx1[0]");
     static_assert(are_equal(dummy21.second,com12),"Idx2 dummy Idx1[1]");
     static_assert(are_equal(dummy23.first,com21),"Idx2 dummy Idx3[0]");
     static_assert(are_equal(dummy23.second,dum23),"Idx2 dummy Idx3[1]");
     static_assert(are_equal(dummy32.first,com32),"Idx3 dummy Idx2[0]");
     static_assert(are_equal(dummy32.second,dum32),"Idx3 dummy Idx2[1]");


    return tester.results();
}
