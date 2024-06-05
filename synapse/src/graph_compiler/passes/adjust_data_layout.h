#pragma once

#include "types.h"
#include "transpose_inserter.h"

class DataLayoutHandler
{
public:
    DataLayoutHandler(const HabanaGraph& g, const NodePtr& node);
    bool       validate() const;
    bool       canExtract() const;
    const TransposeNodeParamsVector& extract(HabanaGraph& g);  // used by Eager compilation mode
    bool       extractAndReplace(HabanaGraph& g) const;

private:
    const NodePtr& m_node;
    bool           m_skipLogic = false;
    std::optional<TransposeInserter> m_transposeInserter;
};