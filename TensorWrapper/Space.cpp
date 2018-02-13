#include "Space.hpp"
#include<numeric>

namespace TensorWrapper {

using size_type = typename Space::size_type;
using range_type = typename Space::range_type;

size_type Space::size()const noexcept {
    return  subspaces_.size() ?
            std::accumulate(lengths_.begin(), lengths_.end(), 1L,
            [](const size_type& total, const range_type& val){
                return total*(val[1] - val[0]); //divided by val[2]
            }) : 0L;
}

}
