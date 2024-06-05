#pragma once

#include "code_generator.h"
#include "recipe_generator.h"
#include "gaudi2_types.h"

class Gaudi2CodeGenerator : public CodeGenerator
{
public:
    Gaudi2CodeGenerator(HabanaGraph* graph);
    Gaudi2CodeGenerator(const Gaudi2CodeGenerator& other, HabanaGraph* graph, bool cloneAllocators = false);
    Gaudi2CodeGenerator& operator=(const Gaudi2CodeGenerator& other);
    CodeGeneratorPtr clone(HabanaGraph* graph, bool cloneAllocators = false) const override;

    virtual ~Gaudi2CodeGenerator() = default;

    MemoryAllocator& getWorkspaceAllocator() override { return *m_workspaceAllocator; }
    std::shared_ptr<MemoryAllocator> getWorkspaceAllocatorPtr() override { return m_workspaceAllocator; }
    MemoryAllocator& getAllocatorForProgramData() override { return *m_programAllocator; }
    MemoryAllocator& getSramAllocator() override { return *m_sramAllocator; }
    const MemoryAllocator& getWorkspaceAllocator() const override { return *m_workspaceAllocator; }
    const MemoryAllocator& getAllocatorForProgramData() const override { return *m_programAllocator; }
    const MemoryAllocator& getSramAllocator() const override { return *m_sramAllocator; }
    virtual void           addFinalSyncs(bool bIsActivate = false) override;
    virtual void           fillQueues() override;
    virtual unsigned       getQueueID(HabanaDeviceType type, unsigned id) override;

    virtual void fillQueuesWithDmaNode(NodePtr node) override;
    virtual void setupQueuesMonitors() override;
    virtual void addInitialSyncs(bool bIsActivate = false) override;
    virtual void initQueues() override;
    void         addSFGInitCmd();
    void                                addAllDescriptors() override;  // temporary public
    gaudi2::TpcDescriptorsWrappers&     getTpcNodeDescriptorsWrappers(const NodePtr& n);
    gaudi2::DmaDescriptorsWrappers&     getDmaNodeDescriptorsWrappers(const NodePtr& n);
    gaudi2::MmeDescriptorsWrappers&     getMmeNodeDescriptorsWrappers(const NodePtr& n);
    gaudi2::RotatorDescriptorsWrappers& getRotateNodeDescriptorsWrappers(const NodePtr& n);
    virtual unsigned                    getNumDmaNodeDescriptorsWrappers();
    gaudi2::MmeDescriptorGenerator&     getMmeNodeDescriptorGenerator(const NodePtr& n);
    void setMmeNodeDescriptorGenerator(const NodePtr& n, gaudi2::MmeDescriptorGeneratorPtr& descGenerator);
    void updateTPCDescriptorWrapper(const TPCNode&                       node,
                                    const gaudi2::TpcDesc&               tpcDescriptor,
                                    const ValidityMask<gaudi2::TpcDesc>& tpcMask,
                                    const tpc_wd_ctxt_t&                 tpcFwCtx,
                                    NodeROI&                             roi);

    void updateDMADescriptorWrapper(const DMANode&                       node,
                                    const gaudi2::DmaDesc&               dmaDescriptor,
                                    const ValidityMask<gaudi2::DmaDesc>& dmaMask,
                                    const edma_wd_ctxt_t&                dmaFwCtx,
                                    NodeROI&                             roi);

    void updateRotatorDescriptorWrapper(const RotateNode&                        node,
                                        const gaudi2::RotatorDesc&               rotatorDescriptor,
                                        const ValidityMask<gaudi2::RotatorDesc>& rotateMask,
                                        const rot_wd_ctxt_t&                     rotFwCtx,
                                        NodeROI&                                 roi);

    void updateMmeNodeDescriptorWrapper(const MmeNode& node, const gaudi2::MmeDesc& desc, NodeROI& roi);

private:
    void                         generateRecipes(const HabanaGraph& graph) override;
    virtual void                 initAllocators() override;
    virtual recipe_t*            serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    virtual shape_plane_graph_t* serializeShapePlane(RecipeAllocator* recipeAlloc) const override;

private:
    std::shared_ptr<MemoryAllocator> m_workspaceAllocator;  // for non-persistent tensors
    // program commands and data (aux tensors, kernels and coeff tables)
    std::unique_ptr<MemoryAllocator> m_programAllocator;
    std::unique_ptr<MemoryAllocator> m_sramAllocator;
    std::unique_ptr<RecipeGenerator> m_recipeGenerator;

    std::unordered_map<NodePtr, gaudi2::MmeDescriptorsWrappers>     m_mmeNodesDescriptorsWrappers;
    std::unordered_map<NodePtr, gaudi2::TpcDescriptorsWrappers>     m_tpcNodesDescriptorsWrappers;
    std::unordered_map<NodePtr, gaudi2::DmaDescriptorsWrappers>     m_dmaNodesDescriptorsWrappers;
    std::unordered_map<NodePtr, gaudi2::RotatorDescriptorsWrappers> m_rotNodesDescriptorsWrappers;
    std::unordered_map<NodePtr, gaudi2::MmeDescriptorGeneratorPtr>  m_mmeNodesDescriptorGenerator;
};
