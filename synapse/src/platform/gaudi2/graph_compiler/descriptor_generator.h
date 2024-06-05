#pragma once

#include "gaudi2_code_generator.h"
#include "types.h"
#include "node_visitor.h"
#include "gaudi2_types.h"
#include "node_roi.h"
#include "tpc_node.h"
#include "tpc_slice.h"
#include "tpc_slice_desc_update.h"
#include "utils.h"
#include "sync/sync_scheme_fw_context.h"

class Gaudi2CodeGenerator;

typedef std::array<TensorPtr, MmeCommon::MmeAuxTensorIdx::AUX_TENSOR_MAX_NUM> AuxTensorArray;

namespace gaudi2
{

class DescriptorGenerator : public NodeVisitor
{
public:
    using DmaDescEntry = struct
    {
        gaudi2::DmaDesc desc;
        ValidityMask<gaudi2::DmaDesc> mask;
        edma_wd_ctxt_t  fwCtx;
    };
    using DmaDescriptorsList = std::list<DmaDescEntry>;

    using RotDescEntry = struct
    {
        gaudi2::RotatorDesc desc;
        ValidityMask<gaudi2::RotatorDesc> mask;
        rot_wd_ctxt_t       fwCtx;
    };
    using RotDescriptorsList = std::list<RotDescEntry>;

    DescriptorGenerator(Gaudi2CodeGenerator* codeGenerator);

    void visit(TPCNode* node) override;
    void visit(TPCSlice* node) override;
    void visit(DMANode* node) override;
    void visit(MmeNode* node) override;
    void visit(RotateNode* node) override;

    static void generateDmaDescriptors(const DMANode&            node,
                                       const std::list<NodeROI>& physicalRois,
                                       DmaDescriptorsList&       descriptors);

    static void generateRotatorDescriptors(const RotateNode&         node,
                                           const std::list<NodeROI>& physicalRois,
                                           uint64_t                  sramBase,
                                           RotDescriptorsList&       descriptors);

    static unsigned getRotateStripeWidth(std::shared_ptr<RotateNode>& rotateNode);

    static void generateRotatorEmptyJobDescriptor(RotatorDesc&               desc,
                                                  ValidityMask<RotatorDesc>& descMask,
                                                  uint64_t                   inImgAddr,
                                                  uint64_t                   outImgAddr);

    static void getTensorRoles(const MmeNode&        node,
                               MmeCommon::EMmeOpType opType,
                               TensorPtr&            xTensor,
                               TensorPtr&            wTensor,
                               TensorPtr&            yTensor,
                               TensorPtr&            oTensor,
                               TensorPtr&            aMaskTensor,
                               TensorPtr&            bMaskTensor);

    static void getInputOutputTensors(const MmeNode& node,
                                      TensorPtr&     aTensor,
                                      TensorPtr&     bTensor,
                                      TensorPtr&     cTensor,
                                      TensorPtr&     oTensor,
                                      AuxTensorArray& auxTensors);

    static MmeCommon::EMmeOperand   inputPort2Operand(const MmeCommon::EMmeOpType operation, const bool isA);
    static MmeCommon::EMmeOperand   outputPort2Operand(const MmeCommon::EMmeOpType operation);
    static void                     addDmaCostInfo(DMANode* node);

private:
    void addTpcDescriptorsToGraph(const TPCNode& node, const TPCSliceDescUpdate* updater = nullptr);
    void addDmaDescriptorsToGraph(const DMANode& node);
    void updateMmeDescriptorWrappers(const MmeNode& node);

    void addRotatorDescriptorsToGraph(const RotateNode& node);

    Gaudi2CodeGenerator* m_codeGenerator;
    SyncSchemeFwContext  m_syncSchemeFwContext;
};

}  // namespace gaudi2
