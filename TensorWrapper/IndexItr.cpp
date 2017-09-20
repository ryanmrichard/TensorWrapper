#include "TensorWrapper/IndexItr.hpp"

using namespace TWrapper;
using namespace TWrapper::detail_;

template void next<0>(std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      bool row_major);

template void next<1>(std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      bool row_major);
template void next<2>(std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      bool row_major);
template void next<3>(std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      bool row_major);
template void next<4>(std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      const std::array<size_t,rank>&,
                      bool row_major);

template class IndexItr<0>;
template class IndexItr<1>;
template class IndexItr<2>;
template class IndexItr<3>;
template class IndexItr<4>;
