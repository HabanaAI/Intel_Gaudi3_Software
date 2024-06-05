#pragma once

#include "gaudi3_types.h"
#include "code_generator.h"
#include "recipe_generator.h"

class Gaudi3CodeGenerator : public CodeGenerator
{
public:
    Gaudi3CodeGenerator(HabanaGraph* graph);
    Gaudi3CodeGenerator(const Gaudi3CodeGenerator& other, HabanaGraph* graph, bool cloneAllocators = false);
    Gaudi3CodeGenerator& operator=(const Gaudi3CodeGenerator& other);
    CodeGeneratorPtr clone(HabanaGraph* graph, bool cloneAllocators = false) const override;

    virtual ~Gaudi3CodeGenerator() = default;

    MemoryAllocator& getWorkspaceAllocator() override { return *m_workspaceAllocator; }
    std::shared_ptr<MemoryAllocator> getWorkspaceAllocatorPtr() override { return m_workspaceAllocator; }
    MemoryAllocator& getAllocatorForProgramData() override { return *m_programAllocator; }
    MemoryAllocator& getSramAllocator() override { return *m_sramAllocator; }
    const MemoryAllocator& getWorkspaceAllocator() const override { return *m_workspaceAllocator; }
    const MemoryAllocator& getAllocatorForProgramData() const override { return *m_programAllocator; }
    const MemoryAllocator& getSramAllocator() const override { return *m_sramAllocator; }

    virtual void generateCmeCommands(const NodePtr& n) override;
    inline void  archiveOverlap(std::unique_ptr<gaudi3::Overlap> o) { m_overlapArchive.emplace_back(std::move(o)); }

    // Remove redundant dependencies from the input map using the overlap data-base identified by overlapId
    // A redundant dependency is a dependency that is already satisfied by another dependency in the map
    void removeRedundantDependencies(DependencyMap& depMap, unsigned overlapId) const;

private:
    virtual void      initAllocators() override;
    virtual void      initQueues() override;
    virtual void      fillQueuesWithDmaNode(NodePtr node) override {}
    virtual void      addExecuteDMANode(NodePtr n, uint32_t* inputDmaInd, uint32_t* outputDmaInd) override;
    virtual void      fillQueues() override;
    virtual unsigned  getQueueID(HabanaDeviceType type, unsigned id) override;
    virtual recipe_t* serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    void              generateRecipes(const HabanaGraph& graph) override;

    virtual shape_plane_graph_t*  serializeShapePlane(RecipeAllocator* recipeAlloc) const override;

private:
    std::shared_ptr<MemoryAllocator>              m_workspaceAllocator;
    std::unique_ptr<MemoryAllocator>              m_programAllocator;
    std::unique_ptr<MemoryAllocator>              m_sramAllocator;
    std::unique_ptr<RecipeGenerator>              m_recipeGenerator;
    std::vector<std::unique_ptr<gaudi3::Overlap>> m_overlapArchive;
};
