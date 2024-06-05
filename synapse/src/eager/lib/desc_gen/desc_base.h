#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_gen_hal.h"
#include "node_info/eager_node.h"
#include "recipe_gen/recipe_defs.h"
#include "utils/general_defs.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/habana_nodes/node.h"

namespace eager_mode
{
class EagerGraph;

class DescGeneratorBase
{
public:
    virtual ~DescGeneratorBase() = default;  // Designate destructor to be virtual
    virtual void generateWorkDistributionContexts(SyncSchemeFwContextPtrVariant syncSchemeFwContextPtrVariant) = 0;
    virtual deviceAddrOffset getTensorVirtualAddress(unsigned tensorIdx) const                                 = 0;
    virtual const Byte*      getDescRaw(unsigned descIdx) const                                                = 0;
    virtual const Byte*      getWorkDistributionContextRaw(unsigned descIdx) const                             = 0;

    virtual bool     generateDesc() = 0;
    Node&            getNode() const { return *m_node.get<Node>(); }
    const EagerNode& getEagerNode() const { return m_node; }
    ChipType         getChipType() const { return m_chipType; }
    EngineType       getEngineType() const { return m_node.getEngineType(); }
    virtual bool     isDmaNopDescNeeded() const { return false; }

    size_t getDescNr() const { return m_descNr; }
    size_t getActivationNr() const { return m_activationsNr; }
    size_t getLogicalRoiNr() const { return m_logicalRoisNr; }

    // Queries used mainly in recipe creation
    size_t getRequiredWdCtxNr() const { return m_requiredWdCtxNr; }
    size_t getPatchableTensorsNr() const { return m_patchableTensorsNr; }

protected:
    DescGeneratorBase(EagerGraph& graph, const EagerNode& node);

    // For one activation we need one work distribution object only, for multiple we need two or three:
    //  1st: does not signal and can wait for another engine (used by first activation)
    //  2nd: does not wait for anyone and doesn't signal (valid for 3 activations and more)
    //  3rd: does not wait but signal (used by last activation)
    static size_t calcRequiredWdCtxNr(size_t activationsNr) { return (activationsNr <= 2) ? activationsNr : 3; }

protected:
    // Variables to be initialized at constructor
    EagerGraph&      m_graph;
    const ChipType   m_chipType;
    const EagerNode& m_node;

    // Statistics to be initalized after generating descriptors, it's required for recipe creation
    size_t m_descNr          = 0;
    size_t m_activationsNr   = 0;
    size_t m_logicalRoisNr   = 0;
    size_t m_requiredWdCtxNr = 0;
    // The following field can have non-default value for TPC nodes with shape tensors
    size_t m_patchableTensorsNr = 0;  // Number of patchable tensors. By default it's number of inputs and outputs
};

using DescGeneratorBasePtr = std::unique_ptr<DescGeneratorBase>;

}  // namespace eager_mode