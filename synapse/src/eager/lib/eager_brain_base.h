#pragma once

// synapse-internal includes (relative to src/)
#include "graph_compiler/types.h"

// synapse api (relative to include/)"
#include "synapse_common_types.h"

// relative to <mme>/
#include "include/mme_common/mme_common_enum.h"

class ConvBaseNode;
class MmeNode;
class HalReader;

namespace eager_mode
{
class EagerMmeBrainBase
{
public:
    struct SupportedGeometryProperties
    {
        MmeCommon::EMmeGeometry geometry;
        unsigned                width;
        unsigned                height;
    };

    struct GeometryApproximatedCost
    {
        const SupportedGeometryProperties* geometry;
        uint64_t                           cost;
    };

    constexpr EagerMmeBrainBase(uint64_t conv_packing_threshold, unsigned rollUpTime)
    : m_conv_packing_threshold(conv_packing_threshold), m_rollUpTime(rollUpTime)
    {
    }

    bool                     shouldPackConvWeights(const ConvBaseNode&, uint32_t packingFactor) const;
    GeometryApproximatedCost getBestGeometry(unsigned fcdSize, unsigned spatialSize) const;

    static constexpr unsigned DEFAULT_NUM_SUPPORTED_MME_GEOMETRIES = 4;
    using SupportedGeometries =
        llvm_vecsmall::SmallVector<SupportedGeometryProperties, DEFAULT_NUM_SUPPORTED_MME_GEOMETRIES>;
    virtual const SupportedGeometries& getSupportedGeometries() const = 0;
    virtual unsigned                   getTotalMmeSize() const        = 0;  // Returns total elements supported by HW

    bool                        isNodeCandidateForMmeConcurrency(const MmeNode& node, const HalReader& hal) const;
    virtual bool                isConcurrencySupportedForOutputDataType(synDataType outputDataType) const = 0;
    virtual MmeCommon::ChipType getChipType() const                                                       = 0;

protected:
    virtual unsigned getMaxMmeLength() const = 0;  // Returns maximum elements of 4xh height

private:
    const uint64_t m_conv_packing_threshold;  // in cycles
    const unsigned m_rollUpTime;
};

}  // namespace eager_mode
