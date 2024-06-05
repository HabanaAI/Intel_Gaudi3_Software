#pragma once

#include <vector>
#include <list>
#include <map>
#include "gaudi_types.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "infra/settable.h"
#include "shape_node.h"

struct recipe_t;
class  RecipeGenerator;

class GaudiGraph : public HabanaGraph
{
public:

    GaudiGraph();
    GaudiGraph(const GaudiGraph& other, bool cloneAllocators = false, bool keepMappings = false);
    GaudiGraph& operator =(const GaudiGraph& other);
    virtual ~GaudiGraph();

    virtual bool                         compile() override;
    virtual recipe_t*                    serializeDataPlane(RecipeAllocator* recipeAlloc) const override;
    virtual shape_plane_graph_t*         serializeShapePlane(RecipeAllocator* recipeAlloc) const override;

    virtual HabanaGraphPtr clone(bool cloneAllocators = false, bool keepMappings = false) const override;

    virtual synDeviceType getDeviceType() const override { return synDeviceGaudi; }

    virtual bool graphSupports64BitDataTypes() const override;

    HabanaGraphPtr createEmptyGraph() const override;

    bool validateMemorySection(const InternalSectionHandle* section) const override;

    virtual std::vector<uint32_t> getRollupsArray(NodePtr mmeNode) const override;
    void                          updateMmeRollupsArray(const MmeNode& node, unsigned numRollups);

protected:
    void         initGaudiHalDepMembers();

    virtual bool generateExecutionSchedule() const override;

    synDeviceType   m_deviceType;
private:
    void                             addAllPasses();
    void                             PrintTopologyHBMBandwidth() const;

    std::map<pNode, std::vector<uint32_t>> m_rollupsArray;
};
