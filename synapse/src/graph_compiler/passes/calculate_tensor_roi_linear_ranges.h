#pragma once

#include "habana_pass.h"
#include "graph_compiler/node_annotation.h"
#include "include/mme_common/mme_descriptor_generator_base.h"
#include "graph_compiler/types.h"
#include <optional>

struct NodeROI;
class TensorROI;

typedef struct ActivationOverlapRoi
{
    OverlapRoi roiX;
    OverlapRoi roiY;
    OverlapRoi roiW;
    OverlapRoi roiO;
    uint32_t   numSignals;
    MmeCommon::OperandRoles operandRoles;
} ActivationOverlapRoi;

class CalculateTensorROIsLinearRanges
{
public:
    explicit CalculateTensorROIsLinearRanges() {}

    virtual bool apply(HabanaGraph& g) const;

    virtual void calculateMmeLinearRanges(HabanaGraph& g, const pNode& node) const = 0;
    static void                             calculateMemoryRanges(TensorROI& tRoi, const NodePtr& n, bool isInput);
    static void                             printLinearRanges(const TensorROI& tRoi, bool input);
    static std::vector<DataRange<uint64_t>> toLinearRanges(const CyclicDataRange& cr, uint64_t start, uint64_t end);

    // Legacy way to calculate cyclic ranges, based on previously calculated linear ranges
    static void calculateCyclicRangesFromLinearRanges(TensorROI& tRoi);

    // Legacy way to calculate linear ranges
    static void calculateLinearRangesLegacy(TensorROI& tRoi, const NodePtr& n, bool isInput);

protected:
    void calculateForActivation(const ActivationOverlapRoi& activation, const MmeNode& mmeNode, NodeROI& roi) const;
    void resizeOrigRoi(NodeROI& origROI) const;
    void addRoi(NodeROI                     newRoi,
                const ActivationOverlapRoi& activation,
                std::list<NodeROI>&         newRois,
                unsigned&                   pipeLevel,
                bool                        isAux,
                const MmeNode&              mmeNode) const;
    virtual MmeCommon::EMmeOpType getMmeNodeOpType(const MmeNode&(mmeNode)) const;
};
