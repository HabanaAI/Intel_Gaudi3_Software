#pragma once

#include "code_generator.h"
#include "gaudi_types.h"
#include "recipe_generator.h"

class GaudiCodeGenerator : public CodeGenerator
{
public:
    GaudiCodeGenerator(HabanaGraph* graph);
    GaudiCodeGenerator(const GaudiCodeGenerator& other, HabanaGraph* graph, bool cloneAllocators = false);
    GaudiCodeGenerator& operator=(const GaudiCodeGenerator& other);
    CodeGeneratorPtr clone(HabanaGraph* graph, bool cloneAllocators = false) const override;
    virtual ~GaudiCodeGenerator() = default;

private:
    struct PredicateTable
    {
        PredicateTable() : numPreds(0), firstPredVal(0), deviceAddr(0) {}
        PredicateTable(unsigned a, unsigned b, deviceAddrOffset c, std::shared_ptr<char> d)
        : numPreds(a), firstPredVal(b), deviceAddr(c), hostAddr(d)
        {
        }

        unsigned              numPreds;
        unsigned              firstPredVal;
        deviceAddrOffset      deviceAddr;
        std::shared_ptr<char> hostAddr;
    };

public:
    MemoryAllocator& getWorkspaceAllocator() override { return *m_workspaceAllocator; }
    std::shared_ptr<MemoryAllocator> getWorkspaceAllocatorPtr() override { return m_workspaceAllocator; }
    MemoryAllocator& getAllocatorForProgramData() override { return *m_programAllocator; }
    MemoryAllocator& getSramAllocator() override { return *m_sramAllocator; }
    const MemoryAllocator& getWorkspaceAllocator() const override { return *m_workspaceAllocator; }
    const MemoryAllocator& getAllocatorForProgramData() const override { return *m_programAllocator; }
    const MemoryAllocator& getSramAllocator() const override { return *m_sramAllocator; }
    const std::vector<PredicateTable>& getPredicateTables() const { return m_predicateTables; }

    void                                 addAllDescriptors() override;  // temporary public
    gaudi::DMADescriptorsWrappers&       getDMANodeDescriptorsWrappers(const NodePtr& n);
    gaudi::TPCDescriptorsWrappers&       getTPCNodeDescriptorsWrappers(const NodePtr& n);
    gaudi::MmeDescriptorsWrappers&       getMmeNodeDescriptorsWrappers(const NodePtr& n);
    const gaudi::DMADescriptorsWrappers& getDMANodeDescriptorsWrappers(const NodePtr& n) const;
    const gaudi::TPCDescriptorsWrappers& getTPCNodeDescriptorsWrappers(const NodePtr& n) const;
    const gaudi::MmeDescriptorsWrappers& getMmeNodeDescriptorsWrappers(const NodePtr& n) const;
    void updateMmeNodeDescriptorWrapper(const MmeNode& node, const gaudi::MmeDesc& mmeDesciptor, NodeROI& roi);
    void updateTPCDescriptorWrapper(const TPCNode&                      node,
                                    const gaudi::TpcDesc&               tpcDescriptor,
                                    const ValidityMask<gaudi::TpcDesc>& tpcMask,
                                    NodeROI&                            roi);
    void updateDMADescriptorWrapper(const DMANode&                      node,
                                    const gaudi::DmaDesc&               dmaDescriptor,
                                    const ValidityMask<gaudi::DmaDesc>& dmaMask,
                                    NodeROI&                            roi);

private:
    void                         initQueues() override;
    void                         initTPCQueues() override;
    std::vector<QueueCommandPtr> getLoadPredicateCmds(unsigned numPreds, unsigned firstPredVal = 1);
    deviceAddrOffset             getPredTableDeviceAddr(unsigned numPreds, unsigned firstPredVal);
    deviceAddrOffset             createPredicateTable(unsigned numPreds, unsigned firstPredVal);
    virtual void initAllocators() override;
    virtual void fillQueuesWithDmaNode(NodePtr node) override;
    virtual void addExecuteDMANode(NodePtr n, uint32_t* inputDmaInd, uint32_t* outputDmaInd) override;
    virtual void fillQueues() override;
    virtual unsigned getQueueID(HabanaDeviceType type, unsigned id) override;

    virtual recipe_t*            serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    virtual shape_plane_graph_t* serializeShapePlane(RecipeAllocator* recipeAlloc) const override;
    void                         generateRecipes(const HabanaGraph& graph) override;

    std::vector<PredicateTable>      m_predicateTables;
    std::unique_ptr<RecipeGenerator> m_recipeGenerator;
    std::shared_ptr<MemoryAllocator> m_workspaceAllocator;  // for non-persistent tensors
    // program commands and data (aux tensors, kernels and coeff tables)
    std::unique_ptr<MemoryAllocator> m_programAllocator;
    std::unique_ptr<MemoryAllocator> m_sramAllocator;
    // Maps from a convolution node to its descriptors
    std::map<NodePtr, gaudi::MmeDescriptorsWrappers> m_mmeNodesDescriptorsWrappers;
    std::map<NodePtr, gaudi::TPCDescriptorsWrappers> m_tpcNodesDescriptorsWrappers;
    std::map<NodePtr, gaudi::DMADescriptorsWrappers> m_dmaNodesDescriptorsWrappers;
};