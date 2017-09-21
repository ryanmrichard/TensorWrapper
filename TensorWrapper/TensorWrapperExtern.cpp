#include "TensorWrapper/TensorWrapper.hpp"

namespace TWrapper {

template class TensorWrapper<1,double,detail_::TensorTypes::EigenMatrix>;
template class TensorWrapper<2,double,detail_::TensorTypes::EigenMatrix>;
template class TensorWrapper<1,double,detail_::TensorTypes::EigenTensor>;
template class TensorWrapper<2,double,detail_::TensorTypes::EigenTensor>;
template class TensorWrapper<3,double,detail_::TensorTypes::EigenTensor>;
template class TensorWrapper<4,double,detail_::TensorTypes::EigenTensor>;

#ifdef ENABLE_CTF
template class TensorWrapper<1,double,detail_::TensorTypes::CTF>;
template class TensorWrapper<2,double,detail_::TensorTypes::CTF>;
template class TensorWrapper<3,double,detail_::TensorTypes::CTF>;
template class TensorWrapper<4,double,detail_::TensorTypes::CTF>;
#endif

#ifdef ENABLE_GAXX
template class TensorWrapper<1,double,detail_::TensorTypes::GlobalArrays>;
template class TensorWrapper<2,double,detail_::TensorTypes::GlobalArrays>;
template class TensorWrapper<3,double,detail_::TensorTypes::GlobalArrays>;
template class TensorWrapper<4,double,detail_::TensorTypes::GlobalArrays>;
#endif

#ifdef ENABLE_TILEDARRAY
template class TensorWrapper<1,double,detail_::TensorTypes::TiledArray>;
template class TensorWrapper<2,double,detail_::TensorTypes::TiledArray>;
template class TensorWrapper<3,double,detail_::TensorTypes::TiledArray>;
template class TensorWrapper<4,double,detail_::TensorTypes::TiledArray>;
#endif

}
