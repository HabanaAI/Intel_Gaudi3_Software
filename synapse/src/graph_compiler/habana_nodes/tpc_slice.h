#pragma once

#include "node_roi.h"
#include "tpc_node.h"
#include "operation_slice.h"
#include "node_visitor.h"

class TPCSlice
: public TPCNode
, public OperationSlice
{
    DEFINE_VISITOR_METHOD
public:
    using BaseClass = TPCNode;

    explicit TPCSlice(const TPCNode* tpcNode) : TPCNode(*tpcNode), OperationSlice(this) {}
    ~TPCSlice() override = default;

    NodePtr                     clone() const override;
    tpc_lib_api::GlueCodeReturn init(tpc_lib_api::DeviceId   deviceId,
                                     AuxiliaryTensors*       cachedAuxiliaryTensors,
                                     std::optional<uint32_t> kernelUniqueId) override;
    void                        resetInstantiated() override;
    bool                        isSuggestedOptimizationDone() const override;
    // Custom generate ROI method, since the slice only covers a part of the original operation
    NodeROI generateRoi() const override;

protected:
    // Override slice start offset methods to generate work ROI for the small tensor that's attached to the slice node.
    TOffset getSliceOffset(const TensorPtr&                         tensor,
                           unsigned                                 dim,
                           const tpc_lib_api::DimIndexSpaceMapping& dimAccessPattern) const override;
};