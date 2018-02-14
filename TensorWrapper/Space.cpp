#include "Space.hpp"
#include<numeric>

namespace TensorWrapper {

using size_type = typename Space::size_type;
using range_type = typename Space::range_type;

size_type Space::size()const noexcept {
    return  std::accumulate(lengths_.begin(), lengths_.end(),
                            1L, std::multiplies<>());
}

bool Space::count_(const std::vector<size_type>& idx)const noexcept{
    if(idx.size() > order()) return false;
    for(size_type i=0; i < idx.size(); ++i)
        if(idx[i] >= lengths_[i])
            return false;
    return true;
}

bool Space::operator==(const Space& rhs)const noexcept {
    return lengths_  == rhs.lengths_;
}

bool Space::operator<(const Space& rhs)const noexcept {
    return std::lexicographical_compare(lengths_.begin(), lengths_.end(),
                                        rhs.lengths_.begin(), rhs.lengths_
                                                .end());
}

} //End namespace
