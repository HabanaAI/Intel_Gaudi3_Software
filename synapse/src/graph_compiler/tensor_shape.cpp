#include <algorithm>
#include <cstring>

#include "utils.h"

#include "tensor_shape.h"
#include <string.h>
#include "infra/defs.h"
#include "graph_compiler/habana_global_conf.h"

static const NSizeArray ONE_NSIZE_ARRAY = [] {
    NSizeArray arr;
    arr.fill(1);
    return arr;
}();

TensorShape::TensorShape()
{
    m_maxSizes.fill(1);
    m_minSizes.fill(1);
}

TensorShape::TensorShape(unsigned dim, const SizeArray& sizes, const CoordArray& base) : m_dim(dim), m_isSet(true)
{
    m_maxSizes.fill(1);
    m_minSizes.fill(1);
    setSize(sizes);
    setBase(base);
}

TensorShape::TensorShape(unsigned dim,
            const SizeArray& maxSizes,
            const SizeArray& minSizes,
            const CoordArray& base)
        : m_dim(dim), m_isSet(true)
{
    m_maxSizes.fill(1);
    m_minSizes.fill(1);
    setMaxSize(maxSizes.data(), false);
    setMinSize(minSizes.data(), false);
    setBase(base);
    recalculateIsDynamic();
}

TensorShape::TensorShape(unsigned dim, const NSizeArray& sizes, const NCoordArray& base) : m_dim(dim), m_isSet(true)
{
    m_maxSizes.fill(1);
    m_minSizes.fill(1);
    setSize(sizes.data());
    setBase(base.data());
}

TensorShape::TensorShape(unsigned dim, const NSizeArray& maxSizes, const NSizeArray& minSizes, const NCoordArray& base)
: m_dim(dim), m_isSet(true)
{
    m_maxSizes.fill(1);
    m_minSizes.fill(1);
    setMaxSize(maxSizes.data(), false);
    setMinSize(minSizes.data(), false);
    setBase(base.data());
    recalculateIsDynamic();
}

CoordArray TensorShape::getBases() const
{
    CoordArray a = {};
    std::copy(m_bases.begin(), m_bases.begin() + a.size(), a.begin());
    return a;
}

SizeArray TensorShape::getMaxSizes() const
{
    SizeArray a;
    std::copy(m_maxSizes.begin(), m_maxSizes.begin() + a.size(), a.begin());
    return a;
}

SizeArray TensorShape::getMinSizes() const
{
    SizeArray a;
    std::copy(m_minSizes.begin(), m_minSizes.begin() + a.size(), a.data());
    return a;
}

void TensorShape::setDim(unsigned int dim)
{
    HB_ASSERT(dim <= HABANA_DIM_MAX, "dimension is bigger than maximum dimensions");
    if (m_dim > dim)
    {
        auto size = m_dim - dim;
        memcpy(m_maxSizes.data() + dim, ONE_NSIZE_ARRAY.data(), size * sizeof(m_maxSizes[0]));
        memcpy(m_minSizes.data() + dim, ONE_NSIZE_ARRAY.data(), size * sizeof(m_minSizes[0]));
        memset(m_bases.data() + dim, 0, size * sizeof(m_bases[0]));
    }
    m_dim = dim;
    m_isSet = true;
}

void TensorShape::setSize(const TSize* sizes)
{
    if (!isDynamic())
    {
        setMaxSize(sizes, false);
        setMinSize(sizes, false);
    }
    else
    {
        setMaxSize(sizes);
    }
    m_isSet = true;
}

void TensorShape::setMaxSize(const TSize* sizes, bool recalcIsDynamic)
{
    memcpy(m_maxSizes.data(), sizes, m_dim * sizeof(m_maxSizes[0]));
    recalculateHasZeroSize();
    if (recalcIsDynamic) recalculateIsDynamic();
    m_isSet = true;
}

void TensorShape::setMinSize(const TSize* sizes, bool recalcIsDynamic)
{
    if (m_dim > SYN_MAX_TENSOR_DIM && !GCFG_ENABLE_DYNAMIC_SHAPE_IN_HIGH_DIMENSION.value())
    {
        HB_ASSERT(
            !memcmp(m_maxSizes.data() + SYN_MAX_TENSOR_DIM, sizes + SYN_MAX_TENSOR_DIM, (m_dim - SYN_MAX_TENSOR_DIM) * sizeof(m_maxSizes[0])),
            "Dynamic Shapes is supported only for the first 5 dimensions, and the tensor is dynamic beyond it");
    }
    memcpy(m_minSizes.data(), sizes, m_dim * sizeof(m_minSizes[0]));
    if (recalcIsDynamic) recalculateIsDynamic();
    m_isSet = true;
}

void TensorShape::setBase(const int* base)
{
    memcpy(m_bases.data(), base, sizeof(int) * m_dim);
}

void TensorShape::merge(const TensorShape& rhs)
{
    HB_ASSERT(!isDynamic(), "Merge operation is supported only for non dynamic tensors");

    if (empty())
    {
        *this = rhs;
    }
    else if (! rhs.empty())
    {
        HB_ASSERT(m_dim == rhs.m_dim, "dimensions mismatch");
        for (unsigned int dim = 0; dim < m_dim; ++dim)
        {
            TSize maxSize = std::max(m_bases[dim] + m_maxSizes[dim], rhs.m_bases[dim] + rhs.m_maxSizes[dim]);
            m_bases[dim] = std::min(m_bases[dim], rhs.m_bases[dim]);
            m_maxSizes[dim] = maxSize - m_bases[dim];
            m_minSizes[dim] = m_maxSizes[dim];
        }
        recalculateHasZeroSize();
    }
}

TensorShape TensorShape::merge(const std::list<TensorShape>& shapes)
{
    auto shapeIter = shapes.begin();
    TensorShape ret(*shapeIter);
    for (++shapeIter; shapeIter != shapes.end(); ++shapeIter)
    {
        ret.merge(*shapeIter);
    }
    return ret;
}

void TensorShape::recalculateHasZeroSize()
{
    auto endIter  = m_maxSizes.begin() + m_dim;
    m_hasZeroSize = std::find(m_maxSizes.begin(), endIter, 0) != endIter;
}

void TensorShape::recalculateIsDynamic()
{
    m_isDynamic = (memcmp(m_minSizes.data(), m_maxSizes.data(), m_dim * sizeof(m_minSizes.data()[0])) != 0);
}

std::optional<unsigned> TensorShape::getFirstDynamicDimIndex(unsigned startDim) const
{
    if (!isDynamic())
    {
        return std::nullopt;
    }

    for (int i = startDim; i < m_dim; i++)
    {
        if (m_minSizes[i] != m_maxSizes[i])
        {
            return i;
        }
    }
    return std::nullopt;
}

bool TensorShape::operator==(const TensorShape& o) const
{
    return (m_dim == o.m_dim) && (m_maxSizes == o.m_maxSizes) && (m_minSizes == o.m_minSizes) && (m_bases == o.m_bases);
}

bool TensorShape::operator!=(const TensorShape& o) const
{
    return !operator==(o);
}
