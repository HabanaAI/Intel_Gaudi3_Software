#pragma once

#include "include/mme_common/mme_brain.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_access_pattern.h"
#include "synapse_common_types.h"
#include "types.h"
#include "bundle_view.h"

struct SolutionParams
{
    MmeCommon::PerfAttr                perfAttr;              // solution perf and memory attributes
    MmeCommon::MmeSolutionRequirements solutionRequirements;  // solution requirements in BVD indexes
};
using SolutionParamsPtr = std::shared_ptr<SolutionParams>;

struct MmeSolution
{
    MmeSolution()  = default;
    ~MmeSolution() = default;
    MmeSolution(const MmeSolution& other);
    MmeSolution(MmeSolution&& other) = delete;
    MmeSolution& operator=(const MmeSolution& other) = delete;
    MmeSolution& operator=(MmeSolution&& other) = delete;

    std::unordered_map<NodePtr, MmeCommon::MmeBrainSolutionPtr> brainSolution;  // internal representation
    std::unordered_map<NodePtr, SolutionParamsPtr>              QORs;
    std::unordered_map<BundleViewId, uint64_t>                  bvdMultipliers;
    void chooseSolution() const;
};
using MmeSolutionPtr       = std::shared_ptr<MmeSolution>;
using MmeSolutionContainer = std::vector<MmeSolutionPtr>;

class MmeBrainIfc
{
public:
    MmeBrainIfc(const MmeNode& mmeNode, synDeviceType deviceType);

    // Main API functions
    MmeCommon::MmeLayerParams getRecommendedMmeLayerParams(bool isGeoPreferredShort = true);
    void                      getRecommendedStrategy(MmeCommon::MmeLayerParams& params,
                                                     bool                       ignoreTensorAliasing,
                                                     bool                       isGeoPreferredShort);
    void setRecommendedConcurrency();

    // Layered-brain interfaces:

    // Generate strategies for the given node based on previous solutions for other MME nodes in the bundle.
    // Regarding inflate for utilization (IFU) when the solution contains multiple MME nodes:
    // - The MME brain will check if there is a conflict between the new node and previous nodes
    //   (flattening on the same BVDs + sizes of flattened dims are different).
    // - If there is no conflict or the new node requires flattening of different BVDs,
    //   the MME brain may suggest IFU dims for the new node.
    // - In case of a conflict the new node will not have IFU dims in its strategy.
    MmeSolutionContainer generateLayeredBrainStrategies(const NodePtr&                node,
                                                        const BundleViewContainerPtr& bundleViews,
                                                        const MmeSolutionContainer&   previousSolutions);

    // Inflate the solution to improve the utilization of the given node on the BVDs defined as IFU dims
    // in this node's strategy.
    // After a successful inflation, a new solution will be returned.
    // The new solution will include new multipliers for the requested BVDs for inflation and the QORs
    // of the affected nodes will be updated to reflect the new utilization (in case the inflation BVDs
    // are common to multiple nodes - all the relevant QORs will be updated accordingly).
    // If the utilization threshold is nullopt - the MME brain will return the minimal inflation that
    // increases the utilization. Otherwise, it will return a solution with utilization above that threshold.
    // If the inflation failed - a nullptr will be returned.
    MmeSolutionPtr inflateForUtilization(const MmeSolutionPtr&         solutionToInflate,
                                         const NodePtr&                nodeToInflate,
                                         const BundleViewContainerPtr& bundleViews,
                                         const std::optional<float>&   utilizationThreshold);

    // MME operations access pattern
    static MmeCommon::AccessPattern                    generateAccessPattern(const MmeNode*);
    static MmeCommon::LayerSemantics                   getLayerSemantics(const MmeNode*);
    static MmeCommon::LayerSemantics::TensorProperties tensorProperties(const TensorPtr& t);

    // Auxilary API functions
    unsigned                   getRecommendedGeometryConcurrency();
    static MmeCommon::MmeBrainOperationModes getOperationModes();
    static MmeCommon::ChipType getMmeChipType(const synDeviceType deviceType);
    MmeCommon::PerfAttr        getRecommendedConfigMmePerf();
    bool                       opSupportsChoosingCdConcurrency();
    std::vector<unsigned int>                getCDDims();

    // Return dummy MME memcpy params from given sizes and strides
    static MmeCommon::MmeLayerParams getMmeMemcpyLayerParams(const SizeArray sizes, const StrideArray strides);
    // Calculate MME perfAttr from params directly
    static MmeCommon::PerfAttr getMmePerfFromParams(const MmeCommon::MmeLayerParams params);
    static void                              getMmeConvParams(const MmeNode& mmeNode, MmeCommon::MmeConv& op);
    static std::optional<MmeCommon::MmeConv> getMmeConvParams(const MmeNode& mmeNode);
    bool                       isCdDim(unsigned dim, const MmeCommon::MmeLayerParams& params);
    bool                       isCdDim(unsigned dim);

private:
    MmeCommon::MmeLayerParams getMmeLayerBaseParams() const;
    void                      setStrategyFields(MmeCommon::MmeLayerParams& params, bool ignoreTensorAliasing = true);
    void getRecommendedStrategyFromMmeBrain(MmeCommon::MmeLayerParams& params, bool isGeoPreferredShort);
    MmeCommon::MmeLayerParams getRecommendedConcurrency();
    // These fields are initialized to nullptr because upon node creation the chip type is not known yet
    bool isFullyInitialized() const { return (m_chipType.has_value() && m_mmeBrain.has_value()); }

    const MmeNode&                     m_mmeNode;
    std::optional<MmeCommon::ChipType> m_chipType;
    std::optional<MmeCommon::MmeBrain> m_mmeBrain;
};