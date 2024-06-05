#include "bundle_memory_insulator.h"

#include "graph_editor.h"
#include "habana_graph.h"
#include "node_factory.h"
#include <unordered_set>

using namespace gc::layered_brain;

class POCSlicesInsulator : public POCBundleMemorySliceInsulator
{
public:
    POCSlicesInsulator(HabanaGraph& graph) : m_graph(graph) {}
    virtual ~POCSlicesInsulator() = default;

    void insulate(NodePtr node, TensorPtr slice, BundleInsulation* insulation) final override
    {
        setInsulation(insulation);
        ConsumerClassification consumers = classifyConsumers(slice);
        if (requiresInsulation(node, slice, consumers))
        {
            performInsulation(node, slice, consumers);
        }
    }

protected:
    struct ConsumerClassification
    {
        NodeVector internal;
        NodeVector external;
    };

    struct SliceInsulation
    {
        NodePtr   newNode;
        TensorPtr newSlice;
    };

    // Sub-class interface - return whether the tensor requires insulation
    virtual bool
    requiresInsulation(const NodePtr& node, const TensorPtr& tensor, const ConsumerClassification& consumers) const = 0;

    // Sub-class interface - return a logical node where the copy is an alias of the original
    virtual NodePtr createLogicalInsulationNode(const TensorPtr& orig, const TensorPtr& copy) const = 0;

    // Sub-class interface - return some physical node that copies the original into the copy
    virtual NodePtr createPhysicalInsulationNode(const TensorPtr& orig, const TensorPtr& copy) const = 0;

    const BundleInsulation& insulation() const
    {
        HB_ASSERT_PTR(m_insulation);
        return *m_insulation;
    }

    const HabanaGraph& graph() const { return m_graph; }

    bool nodeInBundle(const NodePtr& node) const { return getBundleIndex(node) == m_insulation->bundleIdx; }

private:
    HabanaGraph&      m_graph;
    BundleInsulation* m_insulation;

    void setInsulation(BundleInsulation* insulation)
    {
        HB_ASSERT_PTR(insulation);
        m_insulation = insulation;
    }

    ConsumerClassification classifyConsumers(const TensorPtr& operand) const
    {
        ConsumerClassification classification;
        for (const NodePtr& consumer : m_graph.getTensorConsumers(operand))
        {
            if (nodeInBundle(consumer))
            {
                classification.internal.push_back(consumer);
            }
            else
            {
                classification.external.push_back(consumer);
            }
        }
        return classification;
    }

    SliceInsulation performInsulation(NodePtr node, TensorPtr tensor, const ConsumerClassification& consumers)
    {
        SliceInsulation sliceInsulation                         = createInsulation(tensor);
        sliceInsulation.newNode->getNodeAnnotation().bundleInfo = node->getNodeAnnotation().bundleInfo;

        replaceOrigTensorByCopy(tensor, sliceInsulation.newSlice, node, consumers.internal);
        fixAliasing(tensor, sliceInsulation.newSlice);
        insertInsulation(tensor, sliceInsulation);

        return sliceInsulation;
    }

    SliceInsulation createInsulation(TensorPtr tensor) const
    {
        if (isAliasToInternalSlice(tensor))
        {
            return createPhysicalInsulation(tensor);
        }
        else
        {
            return createLogicalInsulation(tensor);
        };
    }

    bool isAliasToInternalSlice(const TensorPtr& tensor) const
    {
        const TensorSet& slices = insulation().slices;
        return tensor->isAliasedTensor() &&
               std::find(slices.begin(), slices.end(), tensor->getAliasTensor()) != slices.end();
    }

    SliceInsulation createPhysicalInsulation(TensorPtr tensor) const
    {
        // The insulated tensor is an alias of another slice in the bundle. It can't be logically insulated and the
        // new slice will need to be an alias instead of it.
        SliceInsulation si;
        si.newSlice = getCopy(tensor);
        si.newSlice->cloneAliasInfo(tensor);
        tensor->resetAliasing();
        si.newNode = createPhysicalInsulationNode(tensor, si.newSlice);
        return si;
    }

    SliceInsulation createLogicalInsulation(TensorPtr tensor) const
    {
        // The insulated tensor is either not an alias, or an alias to a tensor that is external to the bundle.
        // Either way, the new internal slice will be an alias of the original (per the specification of
        // createLogicalInsulationNode).
        SliceInsulation si;
        si.newSlice  = getCopy(tensor);
        si.newNode   = createLogicalInsulationNode(tensor, si.newSlice);
        auto logical = std::dynamic_pointer_cast<LogicalOpNode>(si.newNode);
        HB_ASSERT_PTR(logical);
        logical->runAndSetLogicalOp();
        HB_ASSERT(si.newSlice->getAliasTensor() == tensor,
                  "Expect the logical operation node to set the new slice as alias to the original tensor.");
        return si;
    }

    TensorPtr getCopy(const TensorPtr& tensor) const
    {
        TensorPtr newSlice = tensor->clone(false, false, false);
        newSlice->setName(fmt::format("{}_as_slice", tensor->getName()));
        return newSlice;
    }

    void replaceOrigTensorByCopy(const TensorPtr&  orig,
                                 const TensorPtr&  copy,
                                 const NodePtr&    node,
                                 const NodeVector& consumers)
    {
        GraphEditor::replaceTensor(m_graph, node, orig, copy);
        for (NodePtr consumer : consumers)
        {
            if (consumer == node) continue;
            GraphEditor::replaceTensor(m_graph, consumer, orig, copy);
            LOG_DEBUG(LB_CACHE_MNGR,
                      "<< Consumer {} input replaced. copy input index: {}",
                      consumer->getNodeName(),
                      consumer->getInputIndexOfTensor(copy));
        }
    }

    void fixAliasing(const TensorPtr& orig, const TensorPtr& copy)
    {
        for (TensorPtr tensor : m_insulation->slices)
        {
            if (tensor == orig) continue;
            if (tensor->getAliasTensor() == orig)
            {
                tensor->updateAliasTensor(copy);
            }
        }
    }

    void insertInsulation(const TensorPtr& slice, const SliceInsulation& sliceInsulation)
    {
        bool res = GraphEditor::addNode(m_graph, sliceInsulation.newNode);
        HB_ASSERT(res == true,
                  "Failed to insulate slice {} from the graph using {} node",
                  slice->getName(),
                  sliceInsulation.newNode->getGUID());
        m_insulation->updatedBundle.push_back(sliceInsulation.newNode);
        m_insulation->slices.insert(sliceInsulation.newSlice);
    }
};

// Insulate a single input slice
class POCInputSlicesInsulator : public POCSlicesInsulator
{
public:
    POCInputSlicesInsulator(HabanaGraph& graph) : POCSlicesInsulator(graph) {}
    virtual ~POCInputSlicesInsulator() = default;

private:
    bool requiresInsulation(const NodePtr&                node,
                            const TensorPtr&              tensor,
                            const ConsumerClassification& consumers) const override
    {
        if (!tensor) return false;
        if (tensor->isShapeTensor()) return false;

        const NodePtr& producer = graph().getTensorProducer(tensor);
        if (producer && nodeInBundle(producer)) return false;  // Intermediate slice.

        if (node->getNodeType() == Node::TYPE_INTERNAL_SPLIT) return false;  // slice already insulated

        LOG_DEBUG(LB_CACHE_MNGR, "Insulator: found input insulation candidate: {}", tensor->getName());
        LOG_DEBUG(LB_CACHE_MNGR, "  >> Producer: {}", producer ? producer->getNodeName() : "<None>");
        for (const auto& cons : consumers.internal)
        {
            LOG_DEBUG(LB_CACHE_MNGR, "  << Internal Consumer: {}", cons->getNodeName());
        }
        for (const auto& cons : consumers.external)
        {
            LOG_DEBUG(LB_CACHE_MNGR, "  << External Consumer: {}", cons->getNodeName());
        }
        return true;
    }

    // Insulation through fork makes the copy an alias of the original and connects them through logical
    // operation.
    NodePtr createLogicalInsulationNode(const TensorPtr& orig, const TensorPtr& copy) const override
    {
        synSplitParams params {};
        NodePtr        split = NodeFactory::createNode({orig},
                                                {copy},
                                                &params,
                                                NodeFactory::splitNodeInternalTypeName,
                                                fmt::format("split_{}", orig->getName()));
        return split;
    }

    // Insulation through fill physically copies the contents of the original.
    NodePtr createPhysicalInsulationNode(const TensorPtr& orig, const TensorPtr& copy) const override
    {
        NodePtr fillNode = NodeFactory::createNode({orig},
                                                   {copy},
                                                   nullptr,
                                                   NodeFactory::memcpyNodeTypeName,
                                                   fmt::format("fill_{}", orig->getName()));
        return fillNode;
    }
};

// Insulate a single output slice
class POCOutputSlicesInsulator : public POCSlicesInsulator
{
public:
    POCOutputSlicesInsulator(HabanaGraph& graph) : POCSlicesInsulator(graph) {}
    virtual ~POCOutputSlicesInsulator() = default;

private:
    bool requiresInsulation(const NodePtr&                node,
                            const TensorPtr&              tensor,
                            const ConsumerClassification& consumers) const override
    {
        if (!tensor) return false;
        if (tensor->isShapeTensor()) return false;

        if (consumers.internal.empty() || consumers.external.empty())
        {
            return false;  // Only need to insulate tensors consumed internally and externally
        }

        LOG_DEBUG(LB_CACHE_MNGR, "Insulator: found output insulation candidate: {}", tensor->getName());
        LOG_DEBUG(LB_CACHE_MNGR, "  >> Producer: {}", node->getNodeName());
        for (const auto& cons : consumers.internal)
        {
            LOG_DEBUG(LB_CACHE_MNGR, "  << Internal Consumer: {}", cons->getNodeName());
        }
        for (const auto& cons : consumers.external)
        {
            LOG_DEBUG(LB_CACHE_MNGR, "  << External Consumer: {}", cons->getNodeName());
        }
        return true;
    }

    NodePtr createLogicalInsulationNode(const TensorPtr& orig, const TensorPtr& copy) const override
    {
        synConcatenateParams params {};
        NodePtr              concat = NodeFactory::createNode({copy},
                                                 {orig},
                                                 &params,
                                                 NodeFactory::concatenateNodeLogicalInternalTypeName,
                                                 fmt::format("concat_{}", orig->getName()));
        return concat;
    }

    // Insulation through spill physically copies the contents of copy into the original.
    NodePtr createPhysicalInsulationNode(const TensorPtr& orig, const TensorPtr& copy) const override
    {
        NodePtr spillNode = NodeFactory::createNode({copy},
                                                    {orig},
                                                    nullptr,
                                                    NodeFactory::memcpyNodeTypeName,
                                                    fmt::format("spill_{}", orig->getName()));
        return spillNode;
    }
};

POCBundleMemoryInsulator::POCBundleMemoryInsulator(HabanaGraph&                   graph,
                                                   const BundleNodes&             bundle,
                                                   POCBundleMemorySliceInsulator* inputsInsulator,
                                                   POCBundleMemorySliceInsulator* outputsInsulator)
: m_graph(graph), m_insulation(bundle), m_inputsInsulator(inputsInsulator), m_outputsInsulator(outputsInsulator)
{
    if (!inputsInsulator)
    {
        m_defaultInputInsulator.reset(new POCInputSlicesInsulator(graph));
        m_inputsInsulator = m_defaultInputInsulator.get();
    }

    if (!outputsInsulator)
    {
        m_defaultOutputInsulator.reset(new POCOutputSlicesInsulator(m_graph));
        m_outputsInsulator = m_defaultOutputInsulator.get();
    }
}

// Given a bundle with slices that are consumed both internally and externally like:
// [in]->n0->[t0]->n1->[out]
//             |
//             +-->extCons
// where n0 and n1 are bundled,
// This method adds a bundled join node between the slices and the BPT:
// [in]->n0->[t0_tile]->n1->[out]
//             |
//             +-->join->[t0]->extCons
// n0, n1 and join are bundled.
// Returned bundle order is n0, join, n1.
//
// Given a bundle with input BPTs that are consumed directly, like:
// [in]->n0->...
// Without going through fork (e.g. [in]->split->[in_slice]->n0->...),
// Then handling of 'in' tensor will have a lot of exception cases in case it is decided to be fetched to SRAM.
// The insulator transform this pattern into the desired:
// [in]->fork->[in_slice]->n0->...
BundleNodes POCBundleMemoryInsulator::getInsulatedBundle()
{
    for (const NodePtr& node : m_insulation.origBundle)
    {
        insulate(node);
    }
    return m_insulation.updatedBundle;
}

// Insulate a single node
void POCBundleMemoryInsulator::insulate(const NodePtr& node)
{
    // First add insulation to any required input. Those insulations should be scheduled before 'node'
    insulateInputs(node);
    // Node should be scheduled next
    m_insulation.updatedBundle.push_back(node);
    // Lastly add insulation to any required output. Those insulations should be scheduled after 'node'
    insulateOutputs(node);
}

void POCBundleMemoryInsulator::insulateInputs(const NodePtr& node)
{
    HB_ASSERT_PTR(m_inputsInsulator);

    // The node may have it's outputs replaced during insulation
    TensorVector inputs = node->getInputs();

    for (const TensorPtr& input : inputs)
    {
        m_inputsInsulator->insulate(node, input, &m_insulation);
    }
}

void POCBundleMemoryInsulator::insulateOutputs(const NodePtr& node)
{
    HB_ASSERT_PTR(m_outputsInsulator);

    // The node may have it's outputs replaced during insulation
    TensorVector outputs = node->getOutputs();

    for (const TensorPtr& output : outputs)
    {
        m_outputsInsulator->insulate(node, output, &m_insulation);
    }
}