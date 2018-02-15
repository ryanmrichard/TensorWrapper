#include "Space.hpp"
#include<UtilitiesEx/IterTools/Zip.hpp>
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
    return std::lexicographical_compare(idx.cbegin(), idx.cend(),
                                        lengths_.cbegin(), lengths_.cend());
}

Space& Space::shuffle_(const std::vector <size_type>& from,
                       const std::vector <size_type>& to)  {
    std::vector<size_type> temp(lengths_);
    for(auto p : UtilitiesEx::Zip(from, to))
        temp[std::get<1>(p)] = lengths_[std::get<0>(p)];
    lengths_.swap(temp);
    return *this;
}

bool Space::operator==(const Space& rhs)const noexcept {
    return lengths_  == rhs.lengths_;
}

bool Space::operator<(const Space& rhs)const noexcept {
    return rhs.count(lengths_) && rhs.count(std::vector<size_type>(order()));
}

} //End namespace
