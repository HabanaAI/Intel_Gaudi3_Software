#pragma once

#include "types.h"
#include "tensor.h"
#include "tpc_node.h"
#include "utils.h"
#include <vector>

using namespace gc::access_pattern;

// begin() and end() and operator++ iterate over all the combinations of slice size and offset in the given dim size and
// granularity. For example, in a dimension of size 20 and granularity 8, the iterator would produce all the
// combinations of slice-size x offset in {8,16,20,4*,12*}x{0,8,16}, s.t offset + slice size <= 20 - the dimension
// size.
// ----
// * If the slice is aligned to the end of the dimension, then it's size may not be a multiple of the granularity.
// For example, in dimension size 20 and offset 16, the slice size would be 4.
class DimSliceRange
{
public:
    DimSliceRange(unsigned dimSize, unsigned granularity) : m_dimSize(dimSize), m_granularity(granularity) {}

    struct DimSlice
    {
        unsigned offset;
        unsigned sliceSize;

        bool operator==(const DimSlice& other) const { return offset == other.offset && sliceSize == other.sliceSize; }
    };

    struct Iterator
    {
        Iterator(unsigned dimSize, unsigned granularity)
        : m_dimSize(dimSize), m_granularity {granularity}, m_currDimSlice {0, granularity}
        {
        }
        Iterator(const Iterator& other) = default;
        Iterator& operator=(const Iterator& other) = default;
        ~Iterator()                                = default;

        Iterator() = delete;

        DimSlice        operator*() const { return m_currDimSlice; }
        const DimSlice& operator->() const { return m_currDimSlice; }

        Iterator& operator++();     // prefix
        Iterator  operator++(int);  // postfix
        bool      operator==(const Iterator& other) const;
        bool      operator!=(const Iterator& other) const { return !(*this == other); }

        static Iterator endIterator(unsigned dimSize, unsigned granularity);

    private:
        unsigned m_dimSize;
        unsigned m_granularity;
        DimSlice m_currDimSlice;

        void advance();
        bool advanceSliceSize();
        void advanceOffset();
        void setInitialSliceSizeForOffset();

        bool ended() const { return m_currDimSlice.sliceSize + m_currDimSlice.offset > m_dimSize; }
    };

    Iterator begin() const { return Iterator(m_dimSize, m_granularity); }
    Iterator end() const { return Iterator::endIterator(m_dimSize, m_granularity); }

private:
    unsigned m_dimSize;
    unsigned m_granularity;
};

// TPC node with specific index space and access pattern that allows testing of the TPC Slice different
// functionalities.
class TPCCustomIndexSpaceNode : public TPCNode
{
public:
    struct Params
    {
        struct DimParams
        {
            DimParams(unsigned dimSize, unsigned dimGranularity, int dimInputOverlap = 0, int dimInputOffset = 0)
            : size(dimSize), granularity(dimGranularity), inputOverlap(dimInputOverlap), inputOffset(dimInputOffset)
            {
            }
            unsigned size;
            unsigned granularity;
            int      inputOverlap;  // overlap between adjacent granules - input only
            int      inputOffset;
        };
        std::vector<DimParams> dims;

        // Output is transposed relative to the input - relevant only for the 2 inner dims.
        bool transpose = false;

        std::string name;
    };

    static NodePtr create(const Params& params, TensorPtr userInput = nullptr, TensorPtr userOutput = nullptr);

    // Create a node with mapping of 1:1 between input and output, with granularity 1 and no overlap,
    // so it can be sliced in any way
    static NodePtr createSliceableNode(TensorPtr userInput, TensorPtr userOutput);

protected:
    TPCCustomIndexSpaceNode(const TensorVector& inputs, const TensorVector& outputs, const Params& params);
    void initIndexSpace();
    void initIndexSpaceGeometry();
    void initInputAccessPattern();
    void initOutputAccessPattern();

private:
    const Params m_params;
    static unsigned m_nodeIndex;
};

// TPC node with specific index-space mapping for each tensor, the mapping allows marking tensor dims as all-required.
class TPCCustomIndexSpaceMappingNode : public TPCNode
{
public:
    // The bool indicates if this tensor dimension is all-required, empty vector indicates that all dims are
    // all-required
    using TensorDimToIndexSpaceDimMapping = std::vector<std::pair<Dim, bool>>;

    struct Params
    {
        unsigned                                     tensorRank;
        unsigned                                     nodeResolutionRank;
        std::vector<TensorDimToIndexSpaceDimMapping> dimMappingForInputs;
        std::vector<TensorDimToIndexSpaceDimMapping> dimMappingForOutputs;
    };

    static NodePtr create(const Params& params);

protected:
    TPCCustomIndexSpaceMappingNode(const TensorVector& inputs, const TensorVector& outputs, const Params& params);
    void initIndexSpace();
    void initTensorAccessPattern(tpc_lib_api::TensorAccessPattern&      tensorAccessPattern,
                                 const TensorDimToIndexSpaceDimMapping& dimMapping);

private:
    const Params              m_params;
    static constexpr TSize    TENSOR_DIM_SIZE     = 128;
    static constexpr unsigned RESOLUTION_DIM_SIZE = 2;
    static unsigned           m_nodeIndex;
};
