#ifndef _TENSOR_SHAPE_H_
#define _TENSOR_SHAPE_H_

#include "types.h"
#include <optional>

class TensorShape
{
private:
    void recalculateIsDynamic();
    void recalculateHasZeroSize();

public:
    TensorShape();
    TensorShape(unsigned dim, const SizeArray& sizes, const CoordArray& base = CoordArray({0}));

    TensorShape(unsigned          dim,
                const SizeArray&  maxSizes,
                const SizeArray&  minSizes,
                const CoordArray& base = CoordArray({0}));

    TensorShape(unsigned dim, const NSizeArray& sizes, const NCoordArray& base = NCoordArray({0}));

    TensorShape(unsigned           dim,
                const NSizeArray&  maxSizes,
                const NSizeArray&  minSizes,
                const NCoordArray& base = NCoordArray({0}));

    unsigned   getDim() const { return m_dim; }
    TSize      getSize(unsigned idx) const { return getMaxSize(idx); };
    int        getBase(unsigned idx) const { return m_bases[idx]; }
    SizeArray  getSizes() const { return getMaxSizes(); };
    CoordArray getBases() const;

    const NSizeArray&  getNSizes() const { return m_maxSizes; }
    const NCoordArray& getNBases() const { return m_bases; }
    const NSizeArray&  getNMinSizes() const { return m_minSizes; }

    bool empty() const { return m_dim == 0; };

    void setDim(unsigned int dim);
    void setSize(const SizeArray& sizes) { setSize(sizes.data()); };
    void setSize(const TSize* sizes);
    void setBase(const CoordArray& base) { setBase(base.data()); };
    void setBase(const int* base);

    bool isDynamic() const { return m_isDynamic; };
    bool isDynamicDim(unsigned& dim) const { return dim < m_dim && m_minSizes[dim] != m_maxSizes[dim]; };

    std::optional<unsigned> getFirstDynamicDimIndex(unsigned startDim = 0) const;
    bool hasZeroSize() const { return m_hasZeroSize; }

    TSize     getMaxSize(unsigned idx) const { return m_maxSizes[idx]; };
    TSize     getMinSize(unsigned idx) const { return m_minSizes[idx]; };
    SizeArray getMaxSizes() const;
    SizeArray getMinSizes() const;

    void setMaxSize(const TSize* sizes, bool recalcIsDynamic = true);
    void setMinSize(const TSize* sizes, bool recalcIsDynamic = true);

    /**
     * expand the shape to contain rhs
     */
    void merge(const TensorShape& rhs);

    static TensorShape merge(const std::list<TensorShape>& shapes);

    bool operator==(const TensorShape& other) const;
    bool operator!=(const TensorShape& other) const;

    bool isSet() { return m_isSet; };

private:
    unsigned    m_dim         = 0;
    bool        m_isDynamic   = false;
    bool        m_isSet       = false;
    bool        m_hasZeroSize = false;
    NSizeArray  m_maxSizes    = {};
    NSizeArray  m_minSizes    = {};
    NCoordArray m_bases       = {};
};

#endif  //_TENSOR_SHAPE_H_