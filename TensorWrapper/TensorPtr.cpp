#include "TensorWrapper/TensorPtr.hpp"

namespace TWrapper {
namespace detail_ {
template class TensorPtr<1,double>;
template class TensorPtr<2,double>;
template class TensorPtr<3,double>;
template class TensorPtr<4,double>;

}}
