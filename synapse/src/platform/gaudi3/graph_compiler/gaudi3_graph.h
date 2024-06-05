#pragma once

#include "habana_nodes.h"
#include "gaudi3_types.h"
#include "habana_graph.h"
#include "platform/gaudi3/graph_compiler/tpc_descriptor_generator.h"

using namespace gaudi3;

class RecipeGenerator;

class Gaudi3Graph : public HabanaGraph
{
public:
    Gaudi3Graph();
    Gaudi3Graph(uint64_t sramSize, uint64_t dramSize);
    Gaudi3Graph(const Gaudi3Graph& other, bool cloneAllocators = false, bool keepMappings = false);
    Gaudi3Graph& operator=(const Gaudi3Graph& other);
    virtual ~Gaudi3Graph();
    virtual HabanaGraphPtr clone(bool cloneAllocators = false, bool keepMappings = false) const override;

    virtual bool compile() override;

    virtual bool graphSupports64BitDataTypes() const override;
    virtual synDeviceType    getDeviceType() const override { return synDeviceGaudi3; }

    bool validateMemorySection(const InternalSectionHandle* section) const override;

    HabanaGraphPtr   createEmptyGraph() const override { return std::make_unique<Gaudi3Graph>(); }

    gaudi3::TpcDescriptorsWrappers&     getTpcNodeDescriptorsWrappers(const NodePtr& n);
    gaudi3::MmeDescriptorsWrappers&     getMmeNodeDescriptorsWrappers(const NodePtr& n);
    gaudi3::RotatorDescriptorsWrappers& getRotateNodeDescriptorsWrappers(const NodePtr& n);
    virtual recipe_t*                   serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    void setMmeNodeDescriptorGenerator(const NodePtr& n, std::shared_ptr<gaudi3::MmeDescriptorGenerator> descGenerator);

    gaudi3::MmeDescriptorGenerator& getMmeNodeDescriptorGenerator(const NodePtr& n);
    void                            updateMmeNodeDescriptorWrapper(const MmeNode& node,
                                                                         const gaudi3::MmeDesc& desc,
                                                                         const McidMmeUsage& mcidMmeUsage,
                                                                         NodeROI& roi,
                                                                         unsigned engineIdx);
    void                            updateTPCDescriptorWrapper(const TPCNode&                        node,
                                                               const gaudi3::TpcDesc&                tpcDescriptor,
                                                               const ValidityMask<gaudi3::TpcDesc>&  tpcMask,
                                                               const gaudi3::TpcFwCtxVector&         tpcFwCtxs,
                                                               NodeROI&                              roi,
                                                               TpcDescriptorGenerator::McidTpcUsage& mcidTpcUsage);

    static bool isMmeSlaveSignalingEnabled() { return false; }

    void updateRotatorDescriptorWrapper(const RotateNode&                        node,
                                        const gaudi3::RotatorDesc&               rotatorDescriptor,
                                        const ValidityMask<gaudi3::RotatorDesc>& rotateMask,
                                        const rot_wd_ctxt_t&                     rotFwCtx,
                                        NodeROI&                                 roi);

    unsigned getRotateStripeWidth(std::shared_ptr<RotateNode>& rotateNode) const override;

    virtual void setLogicalQueueToMaxExecutionIndex() override;

    virtual shape_plane_graph_t* serializeShapePlane(RecipeAllocator* recipeAlloc) const override;

protected:
    virtual bool generateExecutionSchedule() const override;
    void         addAllPasses();
    void         addAllDescriptors();
    bool         isValidConfig() const;

    std::unordered_map<NodePtr, gaudi3::MmeDescriptorsWrappers>     m_mmeNodesDescriptorsWrappers;
    std::unordered_map<NodePtr, gaudi3::TpcDescriptorsWrappers>     m_tpcNodesDescriptorsWrappers;
    std::unordered_map<NodePtr, gaudi3::RotatorDescriptorsWrappers> m_rotNodesDescriptorsWrappers;
    std::unordered_map<NodePtr, gaudi3::MmeDescriptorGeneratorPtr>  m_mmeNodesDescriptorGenerator;
};
