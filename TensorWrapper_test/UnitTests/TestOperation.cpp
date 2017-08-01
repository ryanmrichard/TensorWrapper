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
   constexpr static size_t rank=2;
   const Eigen::MatrixXd& tensor_;

   constexpr Convert(const Eigen::MatrixXd& tensor):
       tensor_(tensor)
   {}

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

    //Convert Op (use Eigen::Matrix3d to test the primary template)
    Eigen::Vector3d value2;
    Convert<Eigen::Vector3d> convert(value2);
    Convert<const Eigen::Vector3d> cconvert(value2);
    auto& held_value=convert.eval<type>();
    tester.test("Convert eval",&held_value==&value2);
    const auto& cheld_value=cconvert.eval<type>();
    tester.test("const Convert eval",&cheld_value==&value2);

    //TODO: test when a conversion actually happens

    //AddOp
    Convert<Eigen::MatrixXd> mat(value);
    AddOp<Convert<Eigen::MatrixXd>,Convert<Eigen::MatrixXd>> add = mat + mat;
    Eigen::MatrixXd result=add.eval<type>();
    tester.test("Add eval",result==valuex2);

    //ScaleOp
    auto scale=mat*3.2;
    Eigen::MatrixXd result2=scale.eval<type>();
    tester.test("Scale eval",result2==valuex32);

    //SubtractionOp
    auto sub=add-mat;
    Eigen::MatrixXd result3=sub.eval<type>();
    tester.test("Subtraction",result3==value);

    //IndexedTensor
    auto i=make_index("i");
    auto j=make_index("j");
    using Index_t=Indices<decltype(i),decltype(j)>;

    IndexedTensor<Convert<Eigen::MatrixXd>,Index_t> indices(mat);
    tester.test("Indexed Tensor eval",&(indices.eval<type>())==&value);

    //Contraction
    auto contraction=indices*indices;
    double result4=contraction.eval<type>();
    double corr4=value.cwiseProduct(value).sum();
    tester.test("Idx1*Idx1",result4==corr4);


    return tester.results();
}
