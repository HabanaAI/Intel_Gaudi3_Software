#pragma once

#include <memory>
#include "habana_nodes.h"

struct DmaInsertionPoint
{
    DmaInsertionPoint()
            : t(nullptr)
            , dmaType(DMA_TYPE_UPSTREAM)
    {}

    DmaInsertionPoint(std::shared_ptr<Tensor> t, DMA_TYPE dmaType)
            : t(t)
            , dmaType(dmaType)
    {}

    std::shared_ptr<Tensor> t;
    DMA_TYPE    dmaType;
};

void addDmaNodes(HabanaGraph& g, std::vector<DmaInsertionPoint> points, bool isSetup, const std::string& name);
