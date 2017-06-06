#include <TensorWrapper/Contraction.hpp>
#include <iostream>
#include <Eigen/Dense>
#include "TestHelpers.hpp"

using namespace TWrapper;
using wrapped_type=Eigen::MatrixXd;

const std::array<std::string,3> corr_1({"i","j","k"});
const std::array<std::string,3> corr_2({"ij","jam","klm"});
const std::set<std::string> corr_c1({"i","k"});
const std::map<std::string,size_t> corr_c1_map({{"i",0},{"j",1},
                                                {"k",2},{"l",3}});
const std::set<std::string> corr_c2({"i","k","l"});
const std::map<std::string,size_t> corr_c2_map({{"i",0},{"j",1},
                                                {"k",2},{"l",3},
                                                {"m",4},{"n",5}});
int main()
{
    Tester tester("Testing Einstein notation/Contraction wrapper");

    //Compile time string parsing checks
    detail_::C_String a_string("i,j,k");
    tester.test("C_Str size",a_string.size()==5);
    tester.test("C_Str [0] ",a_string[0]=='i');
    tester.test("C_Str [1] ",a_string[1]==',');
    tester.test("C_Str [2] ",a_string[2]=='j');
    tester.test("C_Str [3] ",a_string[3]==',');
    tester.test("C_Str [4] ",a_string[4]=='k');
    tester.test("C_Str find good ",a_string.find('j',0)==2);
    tester.test("C_Str find multiple ",a_string.find(',',1)==3);
    tester.test("C_Str find bad",a_string.find(',',99)==5);
    auto just_i=a_string.split(',',0);
    tester.test("C_str split works",just_i[0]=='i');
    tester.test("C_str split works",just_i.size()==1);
    auto just_j=a_string.split(',',1);
    tester.test("C_str split works",just_j[0]=='j');
    tester.test("C_str split works",just_j.size()==1);
    auto just_k=a_string.split(',',2);
    tester.test("C_str split works",just_k[0]=='k');
    tester.test("C_str split works",just_k.size()==1);

    Indices<3> idx("i,j,k");
    tester.test("Indices work",idx.idx_[0]==just_i);
    tester.test("Indices work",idx.idx_[1]==just_j);
    tester.test("Indices work",idx.idx_[2]==just_k);

    Eigen::MatrixXd A=Eigen::MatrixXd::Zero(10,10);
    IndexedTensor<3,wrapped_type> syms(A,"i,j,k");
    tester.test("Simple indices",corr_1==syms.idx_);
    IndexedTensor<3,wrapped_type> syms2(A,"ij,jam,klm");
    tester.test("Harder indices",corr_2==syms2.idx_);
    IndexedTensor<3,wrapped_type> syms3(A,"ij , jam , klm");
    tester.test("Harder indices with spaces",corr_2==syms3.idx_);
    IndexedTensor<3,wrapped_type> syms4(A,"ij, jam ,klm");
    tester.test("Harder indices with erratic spacing",corr_2==syms4.idx_);
    IndexedTensor<3,wrapped_type> syms5(A," ij, jam ,klm");
    tester.test("Indices start with space",corr_2==syms5.idx_);
    IndexedTensor<3,wrapped_type> syms6(A," i j , j a m  , k l m ");
    tester.test("Space crazy",corr_2==syms6.idx_);

    IndexedTensor<3,wrapped_type> syms_(A,"l,i,k");
    auto c1=syms*syms_;
    tester.test("Contraction indices",c1.idx2contract_==corr_c1);
    tester.test("Contraction idices map",c1.idx2int_==corr_c1_map);
    IndexedTensor<3,wrapped_type> syms_2(A,"m,l,n");
    auto c2=syms*syms_*syms_2;
    tester.test("Nested contraction indices",c2.idx2contract_==corr_c2);
    tester.test("Nested contraction idices map",c2.idx2int_==corr_c2_map);
    return tester.results();
}
