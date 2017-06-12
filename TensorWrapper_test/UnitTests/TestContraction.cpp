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
    tester.test("Indices constructor",idx.idx_[0]==just_i);
    tester.test("Indices constructor",idx.idx_[1]==just_j);
    tester.test("Indices constructor",idx.idx_[2]==just_k);

    Eigen::MatrixXd A=Eigen::MatrixXd::Zero(10,10);
    IndexedTensor<2,wrapped_type> T1(A,"i,k");
    IndexedTensor<2,wrapped_type> T2(A,"k,l");
    IndexedTensor<2,wrapped_type> T3(A,"l,j");

    auto c1=T1*T2;
    tester.test("NTensors",c1.n_tensors()==2);
    tester.test("Get idx 0 tensor 0",c1.get_index(0,0)==just_i);
    tester.test("Get idx 1 tensor 0",c1.get_index(1,0)==just_k);
    tester.test("Get idx 0 tensor 1",c1.get_index(0,1)==just_k);
    tester.test("position 0 tensor 0",c1.get_position(just_i,0)==0);
    tester.test("position 1 tensor 0",c1.get_position(just_k,0)==1);
    tester.test("position 0 tensor 1",c1.get_position(just_k,1)==0);
    tester.test("NFree",c1.n_free_indices()==2);
    tester.test("NContract",c1.n_contraction_indices()==1);

    auto c2=T1*T2*T3;
    tester.test("NTensors",c2.n_tensors()==3);
    tester.test("Get idx 0 tensor 0",c2.get_index(0,0)==just_i);
    tester.test("Get idx 1 tensor 0",c2.get_index(1,0)==just_k);
    tester.test("Get idx 0 tensor 1",c2.get_index(0,1)==just_k);
    tester.test("Get idx 1 tensor 2",c2.get_index(1,2)==just_j);
    tester.test("position 0 tensor 0",c2.get_position(just_i,0)==0);
    tester.test("position 1 tensor 0",c2.get_position(just_k,0)==1);
    tester.test("position 0 tensor 1",c1.get_position(just_k,1)==0);
    tester.test("position 1 tensor 2",c2.get_position(just_j,2)==1);
    tester.test("NFree",c2.n_free_indices()==2);
    tester.test("NContract",c2.n_contraction_indices()==2);

    return tester.results();
}
