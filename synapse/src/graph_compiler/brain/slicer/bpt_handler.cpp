#include "bpt_handler.h"
#include "node.h"
#include "node_factory.h"
#include "tensor_view_node.h"
#include "tensor_view_shape_node.h"

using namespace gc::layered_brain;

BPTHandler::BPTHandler(const NodePtr& node) : m_bundleIdx(std::nullopt)
{
    for (const auto& input : node->getInputs())
    {
        if (!input) continue;
        m_inputBPTs.emplace(input);
    }

    for (const auto& output : node->getOutputs())
    {
        if (!output) continue;
        m_outputBPTs.emplace(output);
    }
}

BPTHandler::BPTHandler(const HabanaGraph& graph, const BundleIdx bundleIdx, const NodeVector& bundleNodes)
: m_bundleIdx(bundleIdx)
{
    for (const auto& node : bundleNodes)
    {
        for (const auto& input : node->getInputs())
        {
            if (!input) continue;
            const auto& producer = graph.getTensorProducer(input);
            if (!producer || (std::find(bundleNodes.begin(), bundleNodes.end(), producer) == bundleNodes.end()))
            {
                m_inputBPTs.emplace(input);
            }
        }

        for (const auto& output : node->getOutputs())
        {
            if (!output) continue;
            const auto& consumers = graph.getTensorConsumers(output);
            bool        consumedOutsideTheBundle =
                std::any_of(consumers.begin(), consumers.end(), [bundleNodes](const NodePtr& consumer) {
                    return std::find(bundleNodes.begin(), bundleNodes.end(), consumer) == bundleNodes.end();
                });
            if (consumedOutsideTheBundle || output->isUserManagedDram() || consumers.empty())
            {
                m_outputBPTs.emplace(output);
            }
        }
    }
}

bool BPTHandler::isBPT(const TensorPtr& tensor) const
{
    return isInputBPT(tensor) || isOutputBPT(tensor);
}

bool BPTHandler::isInputBPT(const TensorPtr& tensor) const
{
    return (m_inputBPTs.find(tensor) != m_inputBPTs.end());
}

bool BPTHandler::isOutputBPT(const TensorPtr& tensor) const
{
    return (m_outputBPTs.find(tensor) != m_outputBPTs.end());
}

void BPTHandler::addTensorSlice(const TensorPtr&   origTensor,
                                const TensorPtr&   slicedTensor,
                                const OffsetArray& sliceOffset)
{
    HB_ASSERT(isBPT(origTensor), "Expected {} to be a BPT", origTensor->getName());
    if (isInputBPT(origTensor))
    {
        m_forkSlices[origTensor].insert({slicedTensor, sliceOffset});
    }
    else
    {
        m_joinSlices[origTensor].insert({slicedTensor, sliceOffset});
    }
}

NodePtr BPTHandler::createTensorViewNode(const TensorPtr&                   origTensor,
                                         const std::set<TensorSliceOffset>& slicedTensors,
                                         bool                               isRealTensorInput) const
{
    HB_ASSERT(!slicedTensors.empty(), "Expected at least one sliced tensor");

    std::string nodeName = fmt::format("{}_for_tensor_{}{}{}",
                                       isRealTensorInput ? "Fork" : "Join",
                                       origTensor->getName(),
                                       m_bundleIdx.has_value() ? "_bundle_" : "",
                                       m_bundleIdx.has_value() ? std::to_string(*m_bundleIdx) : "");

    std::shared_ptr<TensorViewNode> tensorViewNode;
    if (origTensor->isShapeTensor())
    {
        tensorViewNode = std::make_shared<TensorViewShapeNode>(origTensor, isRealTensorInput, nodeName);
    }
    else
    {
        tensorViewNode = std::make_shared<TensorViewNode>(origTensor, isRealTensorInput, nodeName);
    }

    for (const auto& [slicedTensor, offset] : slicedTensors)
    {
        HB_ASSERT(std::all_of(offset.begin(), offset.end(), [](auto o) { return o >= 0; }), "Expected offset >= 0");
        SizeVector offsets(offset.begin(), offset.begin() + slicedTensor->getDim());
        tensorViewNode->addView(slicedTensor, offsets);
        LOG_DEBUG(LB_SLICER,
                  "Add view for original tensor {}, sliced tensor {} with offsets [{}]",
                  origTensor->getName(),
                  slicedTensor->getName(),
                  toString(offsets, ','));
    }

    if (m_bundleIdx.has_value())
    {
        tensorViewNode->getNodeAnnotation().bundleInfo.set(BundleInfo(*m_bundleIdx, BundleType::UNDEFINED, 0));
    }

    return tensorViewNode;
}

NodeVector BPTHandler::createForkAndJoinNodes() const
{
    NodeVector aggregationNodes;
    for (const auto& [origTensor, slicedTensors] : m_forkSlices)
    {
        LOG_DEBUG(LB_SLICER,
                  "Create Fork node for tensor {} : num of slices {}",
                  origTensor->getName(),
                  slicedTensors.size());
        aggregationNodes.emplace_back(createTensorViewNode(origTensor, slicedTensors, true));
    }

    for (const auto& [origTensor, slicedTensors] : m_joinSlices)
    {
        LOG_DEBUG(LB_SLICER,
                  "Create Join node for tensor {} : num of slices {}",
                  origTensor->getName(),
                  slicedTensors.size());
        aggregationNodes.emplace_back(createTensorViewNode(origTensor, slicedTensors, false));
    }
    return aggregationNodes;
}