#include "mme_access_pattern.h"
#include "index_space_dimensions.h"
#include "spatial_dims_mapping.h"
#include "access_pattern_utils.h"
#include "access_pattern_parallelism.h"

#include <unordered_map>

namespace MmeCommon::AccessPatternDetails
{
using IdxSpcVector = AccessPattern::IndexSpaceVector;
using TensorAP = AccessPattern::TensorAccessPattern;
using OperandMapping = std::unordered_map<EMmeInternalOperand, AccessPattern::IndexSpaceVector>;

namespace Gemm
{
// Builder of index space geometry and access pattern for bgemm operations
class AccessPatternBuilder
{
public:
    explicit AccessPatternBuilder(const LayerSemantics* params) : m_params(params)
    {
        MME_ASSERT(found(Role::X), "No X operand");
        MME_ASSERT(found(Role::W), "No W operand");
        MME_ASSERT(found(Role::Y), "No Y operand");
    }

    AccessPattern build()
    {
        AccessPattern ap;
        ap.indexSpace = createIndexSpace();
        ap.operandAccessPatterns = createAllOperandsAP();

        ap.roleA = Role::X;
        ap.roleB = Role::W;
        ap.roleC = Role::Y;

        return ap;
    }

private:
    using Role = OperandRole;
    using DimMapping = AccessPattern::IndexSpaceVector;
    using OperandMapping = std::unordered_map<Role, DimMapping>;
    using TensorShape = LayerSemantics::TensorProperties::TensorShape;

    const LayerSemantics* m_params;

    IdxSpcVector createIndexSpace() const
    {
        const auto& output = shape(Role::Y);

        IdxSpcVector indexSpace(Gemm::MAX_INDEX_SPACE_DIM, 1);

        indexSpace[DIM_OUT_FCD] = output.at(0);
        indexSpace[DIM_OPERANDS_COMMON] = shape(Role::X).at(xTransposed() ? 1 : 0);
        indexSpace[DIM_OUT_HEIGHT] = output.at(1);
        indexSpace[DIM_BATCH_0] = output.size() > 2 ? output.at(2) : 1;
        indexSpace[DIM_BATCH_1] = output.size() > 3 ? output.at(3) : 1;
        indexSpace[DIM_BATCH_2] = output.size() > 4 ? output.at(4) : 1;
        indexSpace[DIM_BCAST] = 1;
        if (found(Role::MASK_A))
        {
            indexSpace[DIM_MASKS_COMMON] = shape(Role::MASK_A).at(xTransposed() ? 1 : 0);
        }

        return indexSpace;
    }

    AccessPattern::OperandAPMap createAllOperandsAP() const
    {
        AccessPattern::OperandAPMap apMap;
        for (const auto& [role, map] : createAllDimMappings())
        {
            if (found(role))
            {
                MME_ASSERT(map.size() >= rank(role), "Missing dimension mapping for role");
                apMap[role] = createTensorAP(map, shape(role).size());
            }
        }
        return apMap;
    }

    OperandMapping createAllDimMappings() const
    {
        DimMapping xMapping = createDimMapping(xTransposed(), DIM_OPERANDS_COMMON, DIM_OUT_HEIGHT);
        DimMapping wMapping = createDimMapping(wTransposed(), DIM_OUT_FCD, DIM_OPERANDS_COMMON);
        DimMapping yMapping = createDimMapping(DIM_OUT_FCD, DIM_OUT_HEIGHT);
        setBCasts(xMapping, wMapping);

        DimMapping maskAMapping = createDimMapping(xTransposed(), DIM_MASKS_COMMON, DIM_OUT_HEIGHT);
        DimMapping maskBMapping = createDimMapping(wTransposed(), DIM_OUT_FCD, DIM_MASKS_COMMON);
        // There is no known case where the mask_shape[2] > 1, but for compatibility with the legacy (synapse based)
        // generation of the mask access pattern (at least for now):
        auto nonTrivial = [&](Role r, Dim d) { return found(r) && rank(r) > d && shape(r).at(d) > 1; };
        maskAMapping[2] = nonTrivial(Role::MASK_A, 2) ? DIM_NUM_IDENTICAL_MASKS : DIM_BCAST;
        maskBMapping[2] = nonTrivial(Role::MASK_B, 2) ? DIM_NUM_IDENTICAL_MASKS : DIM_BCAST;

        return {{Role::X, xMapping},
                {Role::W, wMapping},
                {Role::Y, yMapping},
                {Role::BIAS, {DIM_OUT_FCD}},
                {Role::MASK_A, maskAMapping},
                {Role::MASK_B, maskBMapping}};
    }

    static DimMapping createDimMapping(bool transposed, Dim dim0Map, Dim dim1Map)
    {
        return createDimMapping(transposed ? dim1Map : dim0Map, transposed ? dim0Map : dim1Map);
    }

    static DimMapping createDimMapping(Dim dim0Map, Dim dim1Map)
    {
        return {dim0Map, dim1Map, DIM_BATCH_0, DIM_BATCH_1, DIM_BATCH_2};
    }

    void setBCasts(DimMapping& xMapping, DimMapping& wMapping) const
    {
        const auto& xSizes = shape(Role::X);
        const auto& wSizes = shape(Role::W);
        const auto& ySizes = shape(Role::Y);

        for (Dim dim = 2; dim < ySizes.size(); dim++)
        {
            auto xSize = xSizes.size() > dim ? xSizes.at(dim) : 1;
            auto wSize = wSizes.size() > dim ? wSizes.at(dim) : 1;

            if (xSize == 1 && wSize > 1)
            {
                xMapping.at(dim) = DIM_BCAST;
            }
            if (xSize > 1 && wSize == 1)
            {
                wMapping.at(dim) = DIM_BCAST;
            }
        }
    }

    static TensorAP createTensorAP(const DimMapping& mapping, size_t rank)
    {
        TensorAP tap {};
        for (size_t dim = 0; dim < rank; dim++)
        {
            tap.dimsAccessPattern.push_back(Utils::create1To1DimAccessPattern(mapping.at(dim)));
        }
        return tap;
    }

    bool found(Role role) const { return m_params->operandShapes.find(role) != m_params->operandShapes.end(); }
    size_t rank(Role role) const { return shape(role).size(); }
    const TensorShape& shape(Role role) const { return m_params->operandShapes.at(role).shape; }
    bool xTransposed() const { return m_params->op == e_mme_atb || m_params->op == e_mme_atbt; }
    bool wTransposed() const { return m_params->op == e_mme_abt || m_params->op == e_mme_atbt; }
};
}  // namespace Gemm

namespace Conv
{
class AccessPatternBuilder
{
public:
    explicit AccessPatternBuilder(const MmeLayerParams* params)
    : m_params(params), m_spDimsMapping(params->conv.spatialDimsNr)
    {
        MME_ASSERT(params, "Null ptr params");
    }

    explicit AccessPatternBuilder(const LayerSemantics* semanticParams)
    : m_semanticParams(semanticParams), m_spDimsMapping(semanticParams->convParams->spatialDimsNr)
    {
        MME_ASSERT(semanticParams, "Null ptr params");
    }

    AccessPattern build() const
    {
        AccessPattern accessPattern {};

        accessPattern.indexSpace = createIndexSpace();

        setXAP(accessPattern);
        setYAP(accessPattern);
        setWAP(accessPattern);

        if (hasOutputCopy()) setOutputCopy(accessPattern);
        if (hasBias()) setBias(accessPattern);
        if (hasShape()) setShape(accessPattern);

        return accessPattern;
    }

private:
    const MmeLayerParams* m_params = nullptr;
    const LayerSemantics* m_semanticParams = nullptr;
    const SpatialDimsMapping m_spDimsMapping;

    using SpatialDim = SpatialDimsMapping::SpatialDim;

    IdxSpcVector createIndexSpace() const
    {
        IdxSpcVector idxSpc(MAX_INDEX_SPACE_DIM, 1);

        idxSpc.at(DIM_IN_CHANNELS) = xSize(0);
        idxSpc.at(DIM_OUT_CHANNELS) = ySize(0);

        m_spDimsMapping.forEachWithBatch(
            [&](auto spDim, const auto& inds) { idxSpc.at(inds.idxSpcDim) = isStrict(spDim) ? ySize(inds.xyDim) : 1; });

        return idxSpc;
    }

    void setXAP(AccessPattern& accessPattern) const
    {
        accessPattern.operandAccessPatterns[OperandRole::X] = createXAP();
        switch (opType())
        {
            case e_mme_fwd:
            case e_mme_dedw:
                accessPattern.roleA = OperandRole::X;
                break;
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                accessPattern.roleC = OperandRole::X;
                break;
            default:
                MME_ASSERT(false, "Unexpected op type");
        }
    }

    TensorAP createXAP() const
    {
        TensorAP tap;
        tap.dimsAccessPattern.push_back(Utils::create1To1DimAccessPattern(DIM_IN_CHANNELS));

        m_spDimsMapping.forEachWithBatch([&](auto spDim, const auto& inds) {
            tap.dimsAccessPattern.push_back(isStrict(spDim)
                                                ? Utils::create1To1DimAccessPattern(inds.idxSpcDim)
                                                : Utils::createAllReqDimAccessPattern(xSize(inds.xyDim), inds.idxSpcDim));
        });

        return tap;
    }

    void setYAP(AccessPattern& accessPattern) const
    {
        accessPattern.operandAccessPatterns[OperandRole::Y] = createYAP();
        switch (opType())
        {
            case e_mme_fwd:
                accessPattern.roleC = OperandRole::Y;
                break;
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                accessPattern.roleA = OperandRole::Y;
                break;
            case e_mme_dedw:
                accessPattern.roleB = OperandRole::Y;
                break;
            default:
                MME_ASSERT(false, "Unexpected op type");
        }
    }

    TensorAP createYAP() const
    {
        TensorAP yap;
        yap.dimsAccessPattern.push_back(Utils::create1To1DimAccessPattern(DIM_OUT_CHANNELS));

        m_spDimsMapping.forEachWithBatch([&](auto spDim, const auto& inds) {
            yap.dimsAccessPattern.push_back(isStrict(spDim)
                                                ? Utils::create1To1DimAccessPattern(inds.idxSpcDim)
                                                : Utils::createAllReqDimAccessPattern(ySize(inds.xyDim), inds.idxSpcDim));
        });

        return yap;
    }

    void setWAP(AccessPattern& accessPattern) const
    {
        accessPattern.operandAccessPatterns[OperandRole::W] = createWAP();

        switch (opType())
        {
            case e_mme_fwd:
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                accessPattern.roleB = OperandRole::W;
                break;
            case e_mme_dedw:
                accessPattern.roleC = OperandRole::W;
                break;
            default:
                MME_ASSERT(false, "Unexpected op type");
        }
    }

    TensorAP createWAP() const
    {
        TensorAP wap;
        wap.dimsAccessPattern.push_back(Utils::create1To1DimAccessPattern(DIM_OUT_CHANNELS));
        wap.dimsAccessPattern.push_back(Utils::create1To1DimAccessPattern(DIM_IN_CHANNELS));

        if (opType() == e_mme_transposed_dedx)
        {
            std::swap(wap.dimsAccessPattern.at(0).indexSpaceDim, wap.dimsAccessPattern.at(1).indexSpaceDim);
        }

        m_spDimsMapping.forEachWithoutBatch([&](auto spDim, const auto& inds) {
            wap.dimsAccessPattern.push_back(Utils::createAllReqDimAccessPattern(wSize(inds.wDim), inds.filterDim));
        });

        return wap;
    }

    bool isStrict(SpatialDim spDim) const
    {
        if (spDim == SpatialDim::BATCH) return true;  // the mapping gives invalid dims, and batch is always strict.

        const auto& conv = convParams();
        auto indices = m_spDimsMapping.getIndices(spDim);
        return wSize(indices.wDim) == 1 && conv.padding.at(Dim(spDim)) == 0 && conv.stride.at(Dim(spDim)) == 1 &&
               xSize(indices.xyDim) == ySize(indices.xyDim);
    }

    bool hasOutputCopy() const { return hasSemRole(OperandRole::OUTPUT_COPY); }

    void setOutputCopy(AccessPattern& ap) const
    {
        ap.operandAccessPatterns[OperandRole::OUTPUT_COPY] = ap.operandAccessPatterns.at(outputRole());
    }

    bool hasBias() const { return hasSemRole(OperandRole::BIAS); }

    void setBias(AccessPattern& ap) const
    {
        ap.operandAccessPatterns[OperandRole::BIAS].dimsAccessPattern.push_back(
            ap.operandAccessPatterns.at(outputRole()).dimsAccessPattern.front());
    }

    bool hasShape() const { return hasSemRole(OperandRole::SHAPE); }

    void setShape(AccessPattern& ap) const
    {
        ap.operandAccessPatterns[OperandRole::SHAPE] = ap.operandAccessPatterns.at(outputRole());
    }

    OperandRole outputRole() const
    {
        switch (opType())
        {
            case e_mme_fwd:
                return OperandRole::Y;
            case e_mme_dedx:
            case e_mme_transposed_dedx:
                return OperandRole::X;
            case e_mme_dedw:
            case e_mme_deterministic_dedw:
                return OperandRole::W;
            default:
                MME_ASSERT(false, "Unexpected op");
                return {};
        }
    }

    bool hasSemRole(OperandRole r) const
    {
        return m_semanticParams != nullptr &&
               m_semanticParams->operandShapes.find(r) != m_semanticParams->operandShapes.end();
    }

    unsigned xSize(const Dim d) const
    {
        return m_params ? m_params->x.sizes.at(d) : m_semanticParams->operandShapes.at(OperandRole::X).shape.at(d);
    }
    unsigned wSize(const Dim d) const
    {
        return m_params ? m_params->w.sizes.at(d) : m_semanticParams->operandShapes.at(OperandRole::W).shape.at(d);
    }
    unsigned ySize(const Dim d) const
    {
        return m_params ? m_params->y.sizes.at(d) : m_semanticParams->operandShapes.at(OperandRole::Y).shape.at(d);
    }

    EMmeOpType opType() const { return m_params ? m_params->opType : m_semanticParams->op; }
    const MmeConv& convParams() const { return m_params ? m_params->conv : *m_semanticParams->convParams; }
};
}  // namespace Conv
}  // namespace MmeCommon::AccessPatternDetails

namespace MmeCommon
{
AccessPattern AccessPatternFactory::createFrom(const MmeLayerParams* params)
{
    MME_ASSERT(params != nullptr, "nullptr params");

    LayerSemantics lsm;
    lsm.op = params->opType;
    lsm.convParams = params->conv;
    lsm.operandShapes[OperandRole::X] = LayerSemantics::TensorProperties {
        LayerSemantics::TensorProperties::TensorShape(params->x.sizes.begin(), params->x.sizes.end())};
    lsm.operandShapes[OperandRole::W] = LayerSemantics::TensorProperties {
        LayerSemantics::TensorProperties::TensorShape(params->w.sizes.begin(), params->w.sizes.end())};
    lsm.operandShapes[OperandRole::Y] = LayerSemantics::TensorProperties {
        LayerSemantics::TensorProperties::TensorShape(params->y.sizes.begin(), params->y.sizes.end())};
    return AccessPatternFactory::createFrom(&lsm);
}

AccessPattern AccessPatternFactory::createFrom(const LayerSemantics* layer)
{
    using namespace AccessPatternDetails;
    MME_ASSERT(layer != nullptr, "nullptr layer parameter");

    AccessPattern accessPattern;
    switch (layer->op)
    {
        case e_mme_fwd:
        case e_mme_dedx:
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
        case e_mme_transposed_dedx:
            accessPattern = Conv::AccessPatternBuilder(layer).build();
            break;
        case e_mme_ab:
        case e_mme_abt:
        case e_mme_atb:
        case e_mme_atbt:
            accessPattern =  Gemm::AccessPatternBuilder(layer).build();
            break;
        case e_mme_gemm_transpose:
        case e_mme_memcpy:
        case e_mme_trans:
        case e_mme_reductionAdd:
            MME_ASSERT(false, "Unsupported operation for access pattern building");
            break;
            // No default on purpose. When new operations are added, compilation will fail unless they are explicitly
            // handled.
    }

    return accessPattern;
}

void AccessPatternFactory::applyParallelism(AccessPattern* accessPattern, size_t idxSpcDim, size_t parallelismLevel)
{
    MME_ASSERT(accessPattern != nullptr, "nullptr access pattern when applying parallelization");

    AccessPatternDetails::AccessPatternParallelismAdjuster adjuster {*accessPattern};
    adjuster.setParallelized(idxSpcDim, parallelismLevel);
}
}  // namespace MmeCommon
