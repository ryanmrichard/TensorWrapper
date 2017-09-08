#include <TensorWrapper/Operations.hpp>
#include <Eigen/Dense>
#include "TestHelpers.hpp"
#include <vector>

/** \file Tests to ensure the Operation class as well as the various
 *  OperationImpl classes are working correctly.
 *
 *
 */

namespace TWrapper {
namespace detail_ {

/* In general Eigen::MatrixXd should be wrapped in a TensorWrapper, but for
 * the purposes of this test we want to bypass testing TensorWrapper.  Thus
 * for this test only we specialize Convert so that it knows about Eigen's
 * matrix class.
 */
template<>
struct Convert<Eigen::MatrixXd> :
        public OperationBase<Convert<Eigen::MatrixXd>>
{
   using scalar_type=double;
   using indices=detail_::Indices<detail_::C_String<'i','\0'>>;
   constexpr static size_t rank=2;
   const Eigen::MatrixXd& tensor_;

   constexpr Convert(const Eigen::MatrixXd& tensor):
       tensor_(tensor)
   {}

   std::array<size_t,2> dimensions()const
   {
       return {(size_t)tensor_.rows(),(size_t)tensor_.cols()};
   }

   template<TensorTypes>
   constexpr const Eigen::MatrixXd& eval()const
   {
       return tensor_;
   }

};

}
}



using namespace TWrapper;
using namespace detail_;

using pTensor=TensorPtr<2,double>;

int main()
{
    Tester tester("Testing Operation class");

    constexpr TensorTypes type=TensorTypes::EigenMatrix;
    Eigen::MatrixXd value=Eigen::MatrixXd::Random(10,10);
    Eigen::MatrixXd valuex2=value+value;
    Eigen::MatrixXd valuex32=value*3.2;

    //IndexedTensor
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");
    using Index_t=Indices<decltype(i),decltype(j)>;
    using Index_t2=Indices<decltype(j),decltype(k)>;
    using Index_t3=Indices<decltype(k),decltype(l)>;
    using Index_t4=Indices<decltype(i),decltype(i)>;

    //AddOp
    Convert<Eigen::MatrixXd> mat(value);
    AddOp<Convert<Eigen::MatrixXd>,Convert<Eigen::MatrixXd>> add = mat + mat;
    Eigen::MatrixXd result=add.eval<type>();
    std::array<size_t,2> corr_dims{10,10};
    tester.test("Add dims",corr_dims==add.dimensions());
    tester.test("Add eval",result==valuex2);

    //ScaleOp
    auto scale=mat*3.2;
    Eigen::MatrixXd result2=scale.eval<type>();
    tester.test("Scale dims",corr_dims==scale.dimensions());
    tester.test("Scale eval",result2==valuex32);

    //SubtractionOp
    auto sub=add-mat;
    Eigen::MatrixXd result3=sub.eval<type>();
    tester.test("Subtraction dims",corr_dims==sub.dimensions());
    tester.test("Subtraction",result3==value);


    IndexedTensor<double,Convert<Eigen::MatrixXd>,Index_t> indices(mat);
    tester.test("Indexed Tensor dims",corr_dims==indices.dimensions());
    tester.test("Indexed Tensor eval",&(indices.eval<type>())==&value);

    //Contraction
    auto contraction=indices*indices;
    double result4=contraction.eval<type>();
    double corr4=value.cwiseProduct(value).sum();
    tester.test("Idx1*Idx1 dims",std::array<size_t,0>{}
                ==contraction.dimensions());
    tester.test("Idx1*Idx1",result4==corr4);


    //Nested Contraction
    IndexedTensor<double,Convert<Eigen::MatrixXd>,Index_t2> indices2(mat);
    IndexedTensor<double,Convert<Eigen::MatrixXd>,Index_t3> indices3(mat);
    Eigen::MatrixXd corr5=value*value*value;
    auto contraction2=indices*indices2*indices3;
    Eigen::MatrixXd result5=contraction2.eval<type>();
    tester.test("Idx1*Idx2*Idx3 dims",corr_dims==contraction2.dimensions());
    tester.test("Idx1*Idx2*Idx3",result5==corr5);

    //Permuation
    Eigen::MatrixXd value2=Eigen::MatrixXd::Random(20,30);
    Convert<Eigen::MatrixXd> mat2(value2);
    Permutation<Index_t,Indices<decltype(j),decltype(i)>,Convert<Eigen::MatrixXd>>
            perm(mat2);
    std::array<size_t,2> corr_idx2{30,20};
    tester.test("Permute dims",corr_idx2==perm.dimensions());
    Eigen::MatrixXd result6=perm.eval<type>();
    tester.test("Transpose",result6==value2.transpose());

    //Trace
    using Tr_t=IndexedTensor<double, Convert<Eigen::MatrixXd>,Index_t4>;
    Tr_t value3(mat2);
    Trace<Tr_t> tr(value3);
    std::array<size_t,0> corr_trdims{};
    tester.test("Trace dims",corr_trdims==tr.dimensions());
    auto result7=tr.eval<type>();
    tester.test("Trace",result7==value2.trace());

    return tester.results();
}
