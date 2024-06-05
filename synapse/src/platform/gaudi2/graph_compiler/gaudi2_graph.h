#pragma once

#include "habana_nodes.h"
#include "gaudi2_types.h"
#include "habana_graph.h"
#include "include/gaudi2/mme_descriptor_generator.h"
#include "gaudi2_types.h"

struct recipe_t;
struct shape_plane_graph_t;
class  RecipeGenerator;

class Gaudi2Graph : public HabanaGraph
{
public:
    Gaudi2Graph();
    Gaudi2Graph(uint64_t sramSize, uint64_t dramSize);
    Gaudi2Graph(const Gaudi2Graph& other, bool cloneAllocators = false, bool keepMappings = false);
    Gaudi2Graph& operator=(const Gaudi2Graph& other);
    virtual ~Gaudi2Graph();

    virtual HabanaGraphPtr clone(bool cloneAllocators = false, bool keepMappings = false) const override;

    virtual bool                        compile() override;
    virtual bool                        graphSupports64BitDataTypes() const override;
    virtual synDeviceType               getDeviceType() const override         { return synDeviceGaudi2; }
    HabanaGraphPtr                      createEmptyGraph() const override;

    virtual recipe_t*                   serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    virtual shape_plane_graph_t*        serializeShapePlane(RecipeAllocator* recipeAlloc) const override;
    bool                                validateMemorySection(const InternalSectionHandle* section) const override;

    // MME slave signaling decision can be taken per node, however, the implementation requires to handle cases
    // where current-node is not available. So meanwhile a global decision for the whole graph is adapted for
    // simplicity and lack of requirements that demand a per-node solution.
    static bool isMmeSlaveSignalingEnabled() { return false; }

    unsigned getRotateStripeWidth(std::shared_ptr<RotateNode>& rotateNode) const override;

protected:
    virtual bool generateExecutionSchedule() const override;
    virtual void initNodeTypeMinPrecision() override;

private:
    void         addAllPasses();
    bool         configHasConflicts() const;
};
