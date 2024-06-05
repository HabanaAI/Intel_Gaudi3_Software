#pragma once

#include "types.h"
#include "node_visitor.h"
#include "gaudi3_types.h"
#include "node_roi.h"
#include "tpc_node.h"
#include "utils.h"
#include "sync/sync_scheme_fw_context.h"
#include "tpc_slice_desc_update.h"

class Gaudi3Graph;

namespace gaudi3
{
typedef std::array<CacheMetaData, MmeCommon::NUM_TENSORS> CacheMetaDataArray;
class DescriptorGenerator : public NodeVisitor
{
public:
    using RotDescEntry = struct
    {
        gaudi3::RotatorDesc desc;
        ValidityMask<gaudi3::RotatorDesc> mask;
        rot_wd_ctxt_t       fwCtx;
    };
    using RotDescriptorsList = std::list<RotDescEntry>;

    DescriptorGenerator(Gaudi3Graph* graph);

    void visit(TPCNode* node) override;
    void visit(MmeNode* node) override;
    void visit(TPCSlice* node) override;
    void visit(RotateNode* node) override;

    static MmeCommon::EMmeOpType getOperationType(const MmeNode& node);

    static void getTensorRoles(const MmeNode&        node,
                               MmeCommon::EMmeOpType opType,
                               TensorPtr&            xTensor,
                               TensorPtr&            wTensor,
                               TensorPtr&            yTensor,
                               TensorPtr&            oTensor);

    static void getTensorCacheMetaData(const MMENodePtr& node, CacheMetaDataArray& cacheMetaDataVec);
    static void getTensorCacheMetaDataForCDParallel(const MMENodePtr& mmeNode, CacheMetaDataArray& cacheMetaDataVec);
    static void getTensorCacheMetaDataFromRoi(const MMENodePtr& node,
                                              const NodeROI&    roi,
                                              CacheMetaData&    aTensorCacheMetaData,
                                              CacheMetaData&    bTensorCacheMetaData,
                                              CacheMetaData&    cTensorCacheMetaData);

    static void getInputOutputTensors(const MmeNode& node,
                                      TensorPtr&     aTensor,
                                      TensorPtr&     bTensor,
                                      TensorPtr&     cTensor,
                                      TensorPtr&     oTensor);

    static MmeCommon::EMmeOperand   inputPort2Operand(const MmeCommon::EMmeOpType operation, const bool isA);
    static MmeCommon::EMmeOperand   outputPort2Operand(const MmeCommon::EMmeOpType operation);

    static void generateRotatorDescriptors(const RotateNode&         node,
                                           const std::list<NodeROI>& physicalRois,
                                           uint64_t                  sramBase,
                                           RotDescriptorsList&       descriptors);

    static unsigned getRotateStripeWidth(std::shared_ptr<RotateNode>& rotateNode);

    static void generateRotatorEmptyJobDescriptor(RotatorDesc&               desc,
                                                  ValidityMask<RotatorDesc>& descMask,
                                                  uint64_t                   inImgAddr,
                                                  uint64_t                   outImgAddr);

private:
    void addTpcDescriptorsToGraph(const TPCNode& node, const TPCSliceDescUpdate* updater = nullptr);
    void updateMmeDescriptorWrappers(const MmeNode& node);
    void addRotatorDescriptorsToGraph(const RotateNode& node);

    Gaudi3Graph* m_graph;
    SyncSchemeFwContext m_syncSchemeFwContext;
};

}  // namespace gaudi3
