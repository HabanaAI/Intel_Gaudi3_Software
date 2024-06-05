#pragma once

#include "graph_compiler/types.h"
#include "graph_compiler/habana_nodes/mme_node.h"
#include "graph_compiler/descriptor_wrapper.h"
#include "graph_compiler/smf/smf_utils.h"

namespace arc_platforms
{

class DynamicMMEPatchPointGeneratorBase
{
    protected:
        int getTensorIndex(const TensorVector& tensorVector, const TensorPtr& tensor);
};

template <typename MmeTypes>
class DynamicMMEPatchPointGenerator : public DynamicMMEPatchPointGeneratorBase
{
protected:
    using MmeDescriptorGenerator = typename MmeTypes::MmeDescriptorGenerator;
    using MmeDesc = typename MmeTypes::MmeDesc;
    using MmeTensorDesc = typename MmeTypes::MmeTensorDesc;
    using MmeCmd = typename MmeTypes::MmeCmd;
    using mme_wd_ctxt_t = typename MmeTypes::mme_wd_ctxt_t;
    using TensorTile = gc::access_pattern::TensorTile;

public:
    DynamicMMEPatchPointGenerator() = default;

    void generateDynamicShapesPatchPoints(const MmeNode&                node,
                                          const MmeDescriptorGenerator& descGenerator,
                                          DescriptorWrapper<MmeDesc>&   descWrapper,
                                          unsigned                      engineIdx);
protected:
    void generateDynamicPatchPointsForOperand(const MmeNode&                node,
                                              const TensorPtr&              tensor,
                                              MmeCommon::EMmeOperand        op,
                                              const MmeDescriptorGenerator& descGenerator,
                                              DescriptorWrapper<MmeDesc>&   descWrapper,
                                              unsigned                      engineIdx);

    void addValidElementsPatchPoint(const MmeNode&              node,
                                    const TensorPtr&            tensor,
                                    const MmeTensorDesc*        tensorDesc,
                                    int                         dim,
                                    DescriptorWrapper<MmeDesc>& descWrapper,
                                    uint64_t                    tileOffset,
                                    uint64_t                    tileSize);

    void addValidElementsPatchPointNoTile(const MmeNode&              node,
                                          const TensorPtr&            tensor,
                                          const MmeTensorDesc*        tensorDesc,
                                          int                         dim,
                                          DescriptorWrapper<MmeDesc>& descWrapper);

    virtual uint32_t getNullDescriptorControlWord() = 0;

    virtual void generateDynamicExecutionPatchPoint(const MmeNode& node, DescriptorWrapper<MmeDesc>& descWrapper) = 0;

    virtual TensorTile getTensorTileFromEngine(const MmeNode&                node,
                                               const TensorPtr&              tensor,
                                               unsigned                      engineIdx,
                                               bool&                         outHaveTile) = 0;
};

}  // namespace gaudi2
