#pragma once

#include "types.h"
#include "defs.h"

namespace gc::access_pattern
{
using Dim       = size_t;            // dimension/rank
using MultiDims = std::vector<Dim>;  // Dims container

template<typename GeometryElementType, typename OffsetElementType>
struct Tile
{
    using Geometry = llvm_vecsmall::SmallVector<GeometryElementType, tpc_lib_api::MAX_TENSOR_DIM>;
    using Offset   = llvm_vecsmall::SmallVector<OffsetElementType, tpc_lib_api::MAX_TENSOR_DIM>;
    using Size     = typename Geometry::value_type;
    using Coord    = typename Offset::value_type;

    // Constructor from geometry and offset
    Tile(const Geometry& g, const Offset& o = {}) : geometry(g), offset(o)
    {
        if (o.empty())
        {
            offset.resize(g.size(), 0);
        }

        HB_ASSERT(offset.size() == geometry.size(),
                  "Geometry and offset rank must be the same, but geometry rank = {}, offset rank = {}",
                  geometry.size(),
                  offset.size());
    }

    // Constructor from (possibly different type and size) geometry and offset containers.
    // This can be used for containers like std::arrays where end() always points to a fixed offset from begin(),
    // regardless of the actual rank.
    template<typename GeometryContainer, typename OffsetContainer>
    Tile(Dim                      rank,
         const GeometryContainer& gCont,
         const OffsetContainer&   oCont,
         // Prevent calling this c'tor for non-iterable types by these dummy params (SFINAE)
         typename GeometryContainer::iterator* = 0,
         typename OffsetContainer::iterator*   = 0)
    : Tile(Geometry(gCont.begin(), std::next(gCont.begin(), rank)),
           Offset(oCont.begin(), std::next(oCont.begin(), rank)))
    {
        // Delegated. Nothing to do
    }

    // Constructor for a simple tile with all sizes and offsets set to the same value
    explicit Tile(Dim rank, Size sizeVal = 1, Coord offsetVal = 0)
    : Tile(Geometry(rank, sizeVal), Offset(rank, offsetVal))
    {
    }

    Geometry geometry;
    Offset   offset;
};

// Tensor tile may have a negative offset in case of operation with padding
typedef Tile<uint64_t, int64_t> TensorTile;
// Node tile may not have negative offset or size
typedef Tile<uint64_t, uint64_t> NodeTile;
// Intersection tile may have negative size due to disjoint tiles intersection.
typedef Tile<int64_t, int64_t> IntersectionTile;

template<typename TileType>
IntersectionTile intersect(const TileType& lhs, const TileType& rhs)
{
    HB_ASSERT(lhs.geometry.size() == rhs.geometry.size(), "Tiles mismatch in intersection");
    HB_ASSERT(lhs.offset.size() == rhs.offset.size(), "Tiles mismatch in intersection");
    HB_ASSERT(lhs.offset.size() == lhs.geometry.size(), "Tiles mismatch in intersection");

    IntersectionTile intersection(lhs.geometry.size(), lhs.geometry, lhs.offset);

    for (Dim dim = 0; dim < rhs.geometry.size(); dim++)
    {
        typename TileType::Coord intersectStart = std::max(lhs.offset[dim], rhs.offset[dim]);
        typename TileType::Coord lhsEnd         = lhs.offset[dim] + lhs.geometry[dim];
        typename TileType::Coord rhsEnd         = rhs.offset[dim] + rhs.geometry[dim];
        typename TileType::Coord intersectEnd   = std::min(lhsEnd, rhsEnd);

        intersection.offset[dim]   = intersectStart;
        intersection.geometry[dim] = intersectEnd - intersectStart;
    }
    return intersection;
}

}  // namespace gc::access_pattern