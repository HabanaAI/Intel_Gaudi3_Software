#include "node_layouts_handler.h"
#include "habana_nodes.h"
#include "transpose_node.h"
#include "squeeze_node.h"
#include <perf_lib_layer_params.h>
#include "fcd_ops_utils.h"
#include "compilation_hal_reader.h"
#include "node_factory.h"

using namespace gc;

NodeLayoutsHandler::NodeLayoutsHandler(HabanaGraph&       graph,
                                       const NodePtr&     node,
                                       Permutation&       permutation,
                                       bool               isInput,
                                       PermutationVector& inputsPermutations,
                                       PermutationVector& outputsPermutations,
                                       bool               isEagerMode /* = false */)
: m_graph(graph),
  m_node(node),
  m_io(node->getNodeIOManager()),
  m_inputPerm(isInput ? permutation : permutation.getInversePermutation()),
  m_outputPerm(m_inputPerm.getInversePermutation()),
  m_isInput(isInput),
  m_inputsPermutations(inputsPermutations),
  m_outputsPermutations(outputsPermutations),
  m_isEagerMode(isEagerMode)
{
}

// aggregation util function
static unsigned calculatePermutedAggregationCost(AggregationNode* node, const Permutation& perm)
{
    unsigned aggregationDim = perm.getIndex(node->getAggregationDim());
    unsigned retVal         = 0;
    for (auto tensor : node->getAliasTensors())
    {
        TSize sizes[Tensor::c_tensorMaxNDim];
        tensor->getAllNSizesInElements(sizes);
        perm.permuteShape(sizes, tensor->getDim());

        unsigned tensorContiguous = tensor->getElementSizeInBytes();
        for (unsigned dim = 0; dim < tensor->getDim(); dim++)
        {
            if (dim <= aggregationDim)
            {
                tensorContiguous *= sizes[dim];
            }
        }
        retVal += FcdOpsUtils::calculateExpectedCost(*tensor,
                                                     CompilationHalReader::getHalReader()->getCacheLineSizeInBytes(),
                                                     tensorContiguous);
    }
    return retVal;
}

// tpc util function
static bool preventLowFCDBufferUtilization(const HabanaGraph& graph, TPCNode* node, const Permutation& perm)
{
    const unsigned tpcVectorSizeInBytes = graph.getHALReader()->getTpcVectorSize();
    TensorPtr      input                = node->getInput(0);
    uint64_t       fcdAfterPermute      = input->getAllNSizesInElements()[perm.getValues().at(0)];

    // broadcasted input case - take the second input
    if (fcdAfterPermute == 1 && node->getNumInputs() == 2)
    {
        input           = node->getInput(1);
        fcdAfterPermute = input->getAllNSizesInElements()[perm.getValues().at(0)];
    }

    const unsigned numOfElementInBuffer = tpcVectorSizeInBytes / input->getElementSizeInBytes();
    float          utilization          = (float)fcdAfterPermute / (float)numOfElementInBuffer;

    return utilization < GCFG_TRANSPOSE_DONT_CARE_MIN_FCD_UTILIZATION_THRESHOLD.value();
}

// squeeze util functions
bool NodeLayoutsHandler::deduceSqueezeAxes(Axes& squeezedAxes)
{
    auto numInputs = m_node->getNumInputs();
    HB_ASSERT(numInputs == 1 || numInputs == 2, "num of inputs must be 1 or 2");
    HB_ASSERT(m_node->getNumOutputs() == 1, "num of outputs must be 1");

    const auto& input  = m_node->getInput(0);
    const auto& output = m_node->getOutput(0);

    unsigned inputDims  = input->getDim();
    unsigned outputDims = output->getDim();
    if (inputDims <= outputDims) return false;

    const NSizeArray& inputShape  = input->getAllNSizesInElements();
    const NSizeArray& outputShape = output->getAllNSizesInElements();
    unsigned          j           = 0;

    for (unsigned i = 0; i < inputDims; i++)
    {
        HB_ASSERT(j <= outputDims,
                  "no reshape has more input dims than output dims while maintaining the conditions below");
        if ((inputShape[i] != outputShape[j] && inputShape[i] == 1) || j == outputDims)
        {
            squeezedAxes.push_back(i);
            continue;
        }
        else if (inputShape[i] == outputShape[j])
        {
            j++;
            continue;
        }
        else
        {
            return false;
        }
    }
    if (j < outputDims) return false;  // didn't go over all the output dims but did all the input dims - not a squeeze
    return true;
}

static Permutation getSqueezedPermutation(const Permutation& perm, const Axes& squeezedAxes, unsigned numDims)
{
    constexpr uint8_t AXIS_TO_REMOVE      = 0xff;
    unsigned          dimsRemovedFromPerm = 0;
    const auto&       permAxes            = perm.getValues();
    auto              squeezedPermAxes    = permAxes;
    for (unsigned dim = 0; dim < numDims; dim++)
    {
        int index = perm.getIndex(dim);
        HB_ASSERT(index >= 0, "all dims must be present in the permutation");
        if (std::find(squeezedAxes.begin(), squeezedAxes.end(), dim) != squeezedAxes.end())
        {
            squeezedPermAxes[index] = AXIS_TO_REMOVE;  // mark to be removed
            dimsRemovedFromPerm++;
        }
        else
        {
            squeezedPermAxes[index] -= dimsRemovedFromPerm;
        }
    }

    squeezedPermAxes.erase(std::remove(squeezedPermAxes.begin(), squeezedPermAxes.end(), AXIS_TO_REMOVE),
                           squeezedPermAxes.end());
    return Permutation(squeezedPermAxes);
}

void NodeLayoutsHandler::fillForSqueezeNode(Axes& squeezedAxes)
{
    // wrapping a squeeze node backwards case - should postpone the attempt
    if (!m_isInput)
    {
        LOG_TRACE(DATA_LAYOUT, "Wrapping a squeeze node backwards case - postpone");
        m_postpone = true;
        return;
    }

    for (auto& axis : squeezedAxes)
    {
        // Permute the squeeze axis
        axis = m_inputPerm.permuteDim(axis);
    }
    unsigned    numInputDims = m_node->getInput(0)->getDim();
    Permutation outputPerm   = getSqueezedPermutation(m_outputPerm, squeezedAxes, numInputDims);

    unsigned numInputs = m_node->getNumInputs();
    m_inputsPermutations.reserve(numInputs);
    m_inputsPermutations.emplace_back(m_inputPerm);
    if (numInputs == 2)
    {
        // In case of shape tensor, the shape represents the output
        m_inputsPermutations.emplace_back(outputPerm.getInversePermutation());
    }
    m_outputsPermutations.emplace_back(outputPerm);
}

// main functions
bool NodeLayoutsHandler::shouldSkip()
{
    if (!m_io.isAllDontCare())
    {
        return true;
    }

    // skip nodes without input/outputs
    if (m_node->getNumInputs() == 0 || m_node->getNumOutputs() == 0)
    {
        return true;
    }

    /* All wrapped node tensors' dimensionality must match the permutation.
     * Exceptions for this are:
     *  1. Scalar tensor.
     *  2. 1D tensor, when the node is not an arithmetic node (i.e., has exactly 2 inputs), which might require
     *     broadcasting later on.
     * Also, the tensors mustn't be of type H2D. */
    unsigned            permSize = m_inputPerm.size();
    const TensorVector& inputs   = m_node->getInputs();
    for (int i = 0; i < inputs.size(); i++)
    {
        const TensorPtr& input = inputs[i];
        if (input == nullptr) continue;
        unsigned dims = input->getDim();

        // TODO: replace with generic code after https://jira.habana-labs.com/browse/SW-164308 is done
        if (input->getTensorType() == HOST_TO_DEVICE_TENSOR && m_node->getNodeType() != Node::eNodeType::TYPE_SLICE)
        {
            LOG_DEBUG(DATA_LAYOUT,
                      "Not wrapping node {} with transposes because it has an H2D tensor",
                      m_node->getNodeName());
            return true;
        }
        if (!GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE.value())
        {
            if (!(dims == permSize || (dims == 1 && (input->getSizeInElements(0) == 1 || m_node->getNumInputs() > 2))))
            {
                LOG_DEBUG(DATA_LAYOUT,
                          "Not wrapping node {} with transposes because num dims mismatch",
                          m_node->getNodeName());
                return true;
            }
        }
    }
    for (const auto& output : m_node->getOutputs())
    {
        if (output == nullptr) continue;
        unsigned dims = output->getDim();
        if (!(dims == permSize && output->getTensorType() != HOST_TO_DEVICE_TENSOR))
        {
            LOG_DEBUG(DATA_LAYOUT,
                      "Not wrapping node {} with transposes because num dims mismatch or H2D tensor",
                      m_node->getNodeName());
            return true;
        }
    }

    return false;
}

void NodeLayoutsHandler::insertReshape(const pTensor& tensorToReshape, unsigned toDims)
{
    std::string name                             = tensorToReshape->getName() + "_reshape";
    TSize       newSizes[Tensor::c_tensorMaxDim] = {0};
    tensorToReshape->getAllSizesInElements(newSizes, Tensor::c_tensorMaxDim);
    HB_ASSERT(tensorToReshape->getDim() < toDims,
              "Trying to reshape a tensor that doesn't have less dims than the reshape dims");
    for (unsigned i = tensorToReshape->getDim(); i < toDims; i++)
    {
        // insert dimensions of size 1 to the right in order to match the given dimensionality
        newSizes[i] = 1;
    }
    // Insert reshape node after input tensor (tensorToReshape)
    TensorPtr newTensor = std::make_shared<Tensor>(toDims, newSizes, tensorToReshape->getElementType());
    newTensor->setName(name);
    newTensor->setAllQuantizationParams(tensorToReshape->getAllQuantizationParams());
    newTensor->setDynamicRange(tensorToReshape->getDynamicRange());

    pNode newNode = NodeFactory::createNode({tensorToReshape}, {newTensor}, nullptr, "reshape", name);

    GraphEditor::replaceTensor(m_graph, m_node, tensorToReshape, newTensor);
    GraphEditor::addNode(m_graph, newNode);

    LOG_DEBUG(GC,
              "Inserted {} node \"{}\" on input \"{}\" of node \"{}\"",
              "reshape",
              name,
              tensorToReshape->getName(),
              m_node->getNodeName());
}

void NodeLayoutsHandler::expandDimsForBroadcast()
{
    const TensorVector& inputs = m_node->getInputs();
    for (int i = 0; i < inputs.size(); i++)
    {
        const TensorPtr& input = inputs[i];
        if (!m_io.isInputRestrictedAtIndex(i) && input->getDim() < m_inputPerm.size())
        {
            insertReshape(input, m_inputPerm.size());
        }
    }
}

void NodeLayoutsHandler::insertTensorsPermutation(bool isInput)
{
    PermutationVector&       tensorsPermutations = isInput ? m_inputsPermutations : m_outputsPermutations;
    const Permutation&       perm                = isInput ? m_inputPerm : m_outputPerm;
    const TensorVector&      tensors             = isInput ? m_node->getInputs() : m_node->getOutputs();
    static const Permutation DONT_CARE_PERM(1);

    tensorsPermutations.reserve(tensors.size());
    for (int i = 0; i < tensors.size(); i++)
    {
        if (tensors[i] == nullptr ||
            (!GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE.value() && tensors[i]->getDim() == 1 && !tensors[i]->isHost2DeviceTensor()) ||
            (isInput ? m_io.isInputRestrictedAtIndex(i) : m_io.isOutputRestrictedAtIndex(i)))
        {
            tensorsPermutations.push_back(DONT_CARE_PERM);
        }
        else
        {
            HB_ASSERT(!tensors[i]->isHost2DeviceTensor() || m_node->getNodeType() == Node::TYPE_SLICE, "Only H2D tensors in Slice nodes can be transposed");
            tensorsPermutations.push_back(perm);
        }
    }
}

void NodeLayoutsHandler::fillTensorsPermutations()
{
    if (shouldSkip()) return;

    // plant reshape nodes on input tensors in case of broadcast inputs
     if (GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE.value() && !m_isEagerMode)
     {
         expandDimsForBroadcast();
     }

    // prepare the TensorsPermutations
    insertTensorsPermutation(true);
    insertTensorsPermutation(false);
}

// visit methods
void NodeLayoutsHandler::visit(Node* node)
{
    fillTensorsPermutations();
}

// TPC node
void NodeLayoutsHandler::visit(TPCNode* node)
{
    const auto& guid = node->getGUID();

    if (!GCFG_ENABLE_RESTRICTED_LAYOUTS_MODE.value() &&
        (startsWith(guid, "reshape") || node->isRestrictedShapeRandomNode()))
    {
        LOG_DEBUG(DATA_LAYOUT, "Node {} is restricted", node->getNodeName());
        return;
    }

    // prevent permute if after the transpose the FCD buffer utilization is too low (WA for SW-142629)
    if (GCFG_TRANSPOSE_DONT_CARE_USE_BFS.value() && (startsWith(guid, "reduce_sum") || startsWith(guid, "mult")))
    {
        if (preventLowFCDBufferUtilization(m_graph, node, m_inputPerm))
        {
            LOG_DEBUG(DATA_LAYOUT,
                      "Not wrapping node {} with transposes to prevent low FCD buffer utilization",
                      node->getNodeName());
            return;
        }
    }

    // prevent permute on logsoftmax if after transpose reduce dim is 0 (WA for SW-91157)
    if (startsWith(guid, "logsoftmax"))
    {
        auto reduceDim = ((ns_Softmax::Params*)node->getParams())->dim;
        if (m_inputPerm.getIndex(reduceDim) == 0)
        {
            LOG_DEBUG(DATA_LAYOUT,
                      "Not wrapping node {} with transposes due to preventPermuteLogSoftMax",
                      node->getNodeName());
            return;
        }
    }

    // prevent propagate transpose backwards through a cast node if it will unnecessarily act on larger data (WA for SW-92752)
    const TensorPtr& input = node->getInput(0);
    if (!m_isEagerMode && !m_isInput && node->isCast() && m_graph.isInputTensor(input) &&
        !input->getTensorAnnotation().memory.allowPermutation &&
        input->getTotalSizeInBytes() > node->getOutput(0)->getTotalSizeInBytes())
    {
        LOG_DEBUG(DATA_LAYOUT,
                  "Not wrapping node {} with transposes due to preventPropagateToLargerData",
                  node->getNodeName());
        return;
    }

    fillTensorsPermutations();
}

// Multi nodes
void NodeLayoutsHandler::visit(TransposeNode* node)
{
    Permutation currPerm(node->permutation());
    if (currPerm == m_inputPerm || currPerm == m_outputPerm)
    {
        LOG_DEBUG(DATA_LAYOUT,
                  "Not wrapping node {} with transposes as it'll forms an identity with the adjacent transpose",
                  m_node->getNodeName());
        return;
    }
    fillTensorsPermutations();
}

void NodeLayoutsHandler::visit(StridedViewNode* node)
{
    LOG_TRACE(DATA_LAYOUT, "Node {} is restricted", m_node->getNodeName());
    return;
}

void NodeLayoutsHandler::visit(StridedInsertNode* node)
{
    LOG_TRACE(DATA_LAYOUT, "Node {} is restricted", m_node->getNodeName());
    return;
}

// MME nodes
void NodeLayoutsHandler::visit(GEMMNode* node)
{
    LOG_TRACE(DATA_LAYOUT, "Node {} is restricted", m_node->getNodeName());
    return;
}

// Logical nodes
void NodeLayoutsHandler::visit(LogicalOpNode* node)
{
    LOG_TRACE(DATA_LAYOUT, "Node {} is restricted", m_node->getNodeName());
    return;
}

void NodeLayoutsHandler::visit(AggregationNode* node)
{
    // Calculate the cost of the Aggregation node using the original layout vs.
    // The cost of the aggregation node when being wrapped with transposes for the user layout.
    // if costInNewLayout > OldLayout * GCFG_TRANSPOSE_DONT_CARE_AGGREGATION_NODE_LOSS_FACTOR
    // Don't wrap with transposes.
    const Permutation identity(node->getOutput(0)->getDim());
    unsigned          defaultCost    = calculatePermutedAggregationCost(node, identity);
    unsigned          transposedCost = calculatePermutedAggregationCost(node, m_inputPerm);
    if (transposedCost >= defaultCost * GCFG_TRANSPOSE_DONT_CARE_AGGREGATION_NODE_LOSS_FACTOR.value())
    {
        LOG_DEBUG(DATA_LAYOUT,
                  "Not wrapping node {} with transposes due to preventPermuteAggregation",
                  m_node->getNodeName());
        return;
    }
    fillTensorsPermutations();
}

void NodeLayoutsHandler::visit(IdentityNode* node)
{
    fillTensorsPermutations();
}

void NodeLayoutsHandler::visit(ReshapeNode* node)
{
    // handle as squeeze node
    Axes squeezedAxes;
    if (deduceSqueezeAxes(squeezedAxes))
    {
        fillForSqueezeNode(squeezedAxes);
        return;
    }

    if (node->isRedundantNode())
    {
        fillTensorsPermutations();
        return;
    }
    LOG_TRACE(DATA_LAYOUT, "Node {} is restricted", m_node->getNodeName());
}

void NodeLayoutsHandler::visit(SqueezeNode* node)
{
    Axes squeezedAxes;
    if (node->isAxisSet())
    {
        squeezedAxes.push_back(node->axisToSqueeze().value());
    }
    else
    {
        deduceSqueezeAxes(squeezedAxes);
    }
    fillForSqueezeNode(squeezedAxes);
}

transposeWrapStatus NodeLayoutsHandler::shouldWrap()
{
    if (m_inputsPermutations.empty() && m_outputsPermutations.empty())
    {
        LOG_TRACE(DATA_LAYOUT, "Shouldn't wrap node {} (postpone={})", m_node->getNodeName(), m_postpone);
        return m_postpone ? transposeWrapPostpone : transposeWrapFail;
    }
    return transposeWrapSuccess;
}
