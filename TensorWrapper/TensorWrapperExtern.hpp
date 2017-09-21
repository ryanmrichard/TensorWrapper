#pragma once

namespace TWrapper {

extern template class TensorWrapper<1,double,detail_::TensorTypes::EigenMatrix>;
extern template class TensorWrapper<2,double,detail_::TensorTypes::EigenMatrix>;
extern template class TensorWrapper<1,double,detail_::TensorTypes::EigenTensor>;
extern template class TensorWrapper<2,double,detail_::TensorTypes::EigenTensor>;
extern template class TensorWrapper<3,double,detail_::TensorTypes::EigenTensor>;
extern template class TensorWrapper<4,double,detail_::TensorTypes::EigenTensor>;

#ifdef ENABLE_CTF
extern template class TensorWrapper<1,double,detail_::TensorTypes::CTF>;
extern template class TensorWrapper<2,double,detail_::TensorTypes::CTF>;
extern template class TensorWrapper<3,double,detail_::TensorTypes::CTF>;
extern template class TensorWrapper<4,double,detail_::TensorTypes::CTF>;
#endif

#ifdef ENABLE_GAXX
extern template class TensorWrapper<1,double,detail_::TensorTypes::GlobalArrays>;
extern template class TensorWrapper<2,double,detail_::TensorTypes::GlobalArrays>;
extern template class TensorWrapper<3,double,detail_::TensorTypes::GlobalArrays>;
extern template class TensorWrapper<4,double,detail_::TensorTypes::GlobalArrays>;
#endif

#ifdef ENABLE_TILEDARRAY
extern template class TensorWrapper<1,double,detail_::TensorTypes::TiledArray>;
extern template class TensorWrapper<2,double,detail_::TensorTypes::TiledArray>;
extern template class TensorWrapper<3,double,detail_::TensorTypes::TiledArray>;
extern template class TensorWrapper<4,double,detail_::TensorTypes::TiledArray>;
#endif

}//End namespace
