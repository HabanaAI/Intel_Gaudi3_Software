#pragma once

#include <array>

// This class encapsulates iteration over multidimensional tensor coordinates.
// WARNING: It's only useful for iterating over dense tensors.
//
// The expected usage:
//
//    const unsigned int bounds[3] = {3, 2, 4};
//    DenseCoordIter<unsigned, 3> it(bounds); // create the iterator
//                                           // starting at {0, 0, 0}
//    it[0] or it[1] or it[2] // access
//    it.next() // advance to {1, 0, 0}
//    it.next() // advance to {2, 0, 0}
//    it.next() // advance to {0, 1, 0}
//    it.next() // advance to {1, 1, 0}
//    it.next() // advance to {2, 1, 0}
//    it.next() // advance to {0, 0, 1}
//    it.next() // advance to {1, 0, 1}
//    it.next() // advance to {2, 0, 1}
//    ...
//    after 3*2*4 calls to it.next(), it.end() will return true.
//
//  You can freeze certain dimensions. The frozen dimensions will not be
//  iterated over. Example:
//
//    const unsigned int bounds[3] = {3, 2, 4};
//    DenseCoordIter<unsigned, 3, DCI::freeze_dims(1)> it(bounds);
//      // create the iterator starting at {0, 0, 0}
//    it[0] or it[1] or it[2] // access
//    // it[1] will always return 0 because that dimension is frozen
//    it.next() // advance to {1, 0, 0}
//    it.next() // advance to {2, 0, 0}
//    it.next() // advance to {0, 0, 1}
//    it.next() // advance to {1, 0, 1}
//    it.next() // advance to {2, 0, 1}
//    it.next() // advance to {0, 0, 2}
//    it.next() // advance to {1, 0, 2}
//    it.next() // advance to {2, 0, 2}
//    it.next() // advance to {0, 0, 3}
//    ...
//    after 3*4 calls to it.next(), it.end() will return true.
//
template<typename T, unsigned int dim, uint32_t frozen_dims_bitmask=0>
class DenseCoordIter
{
public:
    template<typename ArrayLike> DenseCoordIter(ArrayLike bounds);
    // TODO: add a constructor that takes a starting coord when needed
    // NOTE: we could have a (move) constructor that changes the frozen dims

    // Move to the next coordinate
    void next();

    // This is equivalent to calling next() 'n' times
    // next() is more efficient than advance(1)
    void advance(uint64_t n);

    // Returns true when we've iterated past the last element.
    // This iterates over all the coordinates:
    //
    //  for (DenseCoordIter<unsigned, 3> it({2, 3, 4}); !it.end(); ++it)
    //    ...
    //
    bool end() const { return m_end; }

    T operator[](std::size_t idx) const { return m_pos[idx]; }

    unsigned int size() const { return dim; }

    DenseCoordIter<T, dim, frozen_dims_bitmask>& operator++()
    {
        next();
        return *this;
    }

    const T* data() const { return m_pos.data(); }

private:
    // mkBounds is needed to initialize const m_bounds
    template<typename ArrayLike>
    static std::array<T, dim> mkBounds(const ArrayLike arr);

    const std::array<T, dim> m_bounds;
    std::array<T, dim> m_pos;
    bool m_end;
};

namespace DCI  // DenseCoordIter helpers
{
    // Helper for bitmask creation
    constexpr int freeze_dims(int dim) { return 1 << dim; }
    // freeze_dims(0, 1, 2, 4) = 10111 in binary
    template<typename... Args>
    constexpr int freeze_dims(int dim, Args... rest)
    {
        return (1 << dim) | freeze_dims(rest...);
    }
} // namespace DCI

#include  "dense_coord_iter.inl"
