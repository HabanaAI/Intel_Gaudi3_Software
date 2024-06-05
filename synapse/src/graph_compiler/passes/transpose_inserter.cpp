#include "transpose_inserter.h"
#include "node_factory.h"
#include "transpose_node.h"

#include "graph_editor.h"
#include "transpose_utils.h"

using namespace gc;

TransposeInserter::TransposeInserter(const NodePtr&           node,
                                     const PermutationVector& inputPermutations,
                                     const PermutationVector& outputPermutations,
                                     bool                     userTranspose)
: m_node(node),
  m_inputPermutations(inputPermutations),
  m_outputPermutations(outputPermutations),
  m_updateAnnotations(!userTranspose)
{
}

void TransposeInserter::calcExtract(HabanaGraph& g)
{
    // if user requested the transpose, we shouldn't update the annotations that are relevant for data layout optimization passes
    if (m_updateAnnotations)
    {
        m_node->getNodeAnnotation().inputPermutations = m_inputPermutations;
    }

    createTransposesForNodeIO(g, m_inputPermutations, false);
    createTransposesForNodeIO(g, m_outputPermutations, true);
}

const TransposeNodeParamsVector& TransposeInserter::extract(HabanaGraph& g)
{
    calcExtract(g);
    if (!m_origTensorToNewTensor.empty())
    {
        for (const auto& tensors : m_origTensorToNewTensor)
        {
            const auto& origTensor = tensors.first;
            const auto& newTensor  = tensors.second;
            HB_ASSERT(origTensor && newTensor,
                      "Error during transpose insertion for node - tensors to replace are null");
            m_node->replaceTensor(origTensor, newTensor);
        }
    }
    return m_transposesToInsert;
}

bool TransposeInserter::InsertTransposesForNodeIO(HabanaGraph& g)
{
    int retVal = true;
    GraphEditor::editNode(g, m_node, [&]() {
        for (const auto& nodeParams : extract(g))
        {
            NodePtr nodeToAdd = createTransposeNodeFromParams(nodeParams);
            // Maintain tracking of origin nodes for debug purposes
            nodeToAdd->setOriginNodes(m_node->getOriginNodes());
            if (!GraphEditor::addNode(g, nodeToAdd))
            {
                retVal = false;
                break;
            }
        }
    });
    return retVal;
}

void TransposeInserter::createTransposesForNodeIO(HabanaGraph&             g,
                                                  const PermutationVector& permutations,
                                                  bool                     isOutput)
{
    const TensorVector& tensors = isOutput ? m_node->getOutputs() : m_node->getInputs();
    for(std::size_t i = 0; i < permutations.size(); ++i)
    {
        const auto& currentTensor = tensors[i];
        if (currentTensor == nullptr) continue;
        // avoid adding the same transpose twice in case of repeated input tensors
        if (!isOutput && m_origTensorToNewTensor.end() != find_if(m_origTensorToNewTensor.begin(),
                                                                  m_origTensorToNewTensor.end(),
                                                                  [&currentTensor](const auto& tensorPair) {
                                                                      return tensorPair.first == currentTensor;
                                                                  }))
        {
            continue;
        }
        const auto& p = permutations[i];
        if (p.isIdentity()) continue;
        LOG_DEBUG(DATA_LAYOUT,
                  "Inserting transpose for node {}, {}{}, permutation {}",
                  m_node->getNodeName(),
                  isOutput ? "output" : "input",
                  i,
                  p.toString());

        std::string name;
        if (isOutput)
        {
            name = fmt::format("{}_transpose", currentTensor->getName());
        }
        else
        {
            name = fmt::format("{}_input{}_transpose", m_node->getNodeName(), i);
        }
        createTranspose(g, currentTensor, p, isOutput, name);
    }
}

void TransposeInserter::createTranspose(HabanaGraph&       g,
                                        const TensorPtr&   origTensor,
                                        const Permutation& permutation,
                                        bool               isOutput,
                                        const std::string& name)
{
    if (origTensor == nullptr)
    {
        LOG_ERR(DATA_LAYOUT, "Trying to insert transpose {} for a null tensor", name);
        HB_ASSERT(false, "Trying to insert transpose for a null tensor");
        return;
    }

    if (!isOutput && g.getCompilationMode() != CompilationMode::Eager)
    {
        // Optimization: check if requested transpose already exists for the input tensor, and if so use it by changing
        // the node input to be this transpose output tensor, instead of creating a new (identical) transpose
        NodeList consumers = g.getTensorConsumers(origTensor);
        for (const auto& consumer : consumers)
        {
            if (TransposeNode* transpose = dynamic_cast<TransposeNode*>(consumer.get()))
            {
                if (Permutation(transpose->permutation()) == permutation)
                {
                    HB_ASSERT(transpose->getOutputs().size() == 1, "num of outputs must be 1");
                    const TensorPtr& transposeTensor = transpose->getOutput(0);
                    m_origTensorToNewTensor.push_back(std::make_pair(origTensor, transposeTensor));
                    return;
                }
            }
        }
    }

    // allow permutation propagation for Eager
    if (g.getCompilationMode() == CompilationMode::Eager && isOutput && !origTensor->isPersistent() &&
        origTensor->isTrivialStrided())
    {
        origTensor->getTensorAnnotation().memory.allowPermutation = true;
    }

    TransposeNodeParams nodeParams;
    nodeParams.nodeName = name;
    unsigned tensorDim  = origTensor->getDim();
    HB_ASSERT(tensorDim <= MAX_DIMENSIONS_NUM, "invalid tensor size");
    if (permutation.size() != tensorDim && !origTensor->isHost2DeviceTensor())
    {
        LOG_ERR(DATA_LAYOUT,
                "Given permutation of mismatching size {} to tensor of {}-dimensions.",
                permutation.size(),
                tensorDim);
        throw InvalidPermutation();
    }

    // permute new tensor shape
    Permutation shapePermutation = permutation;
    if (isOutput)
    {
        // the node's output tensor, that is being transposed to the given permutation, should have the shape
        // of the inverse permutation.
        shapePermutation = permutation.getInversePermutation();
    }
    const auto&               permValues      = permutation.getValues();
    const auto&               finalPermValues = shapePermutation.getValues();
    TransposePermutationArray permutationArray;
    TensorPtr transposeTensor;
    if (origTensor->isHost2DeviceTensor())
    {
        for (int i = 0; i < permValues.size(); i++)  // always copy all the values (we don't know H2D tensor data rank)
        {
            nodeParams.permutation.push_back((TransposePermutationDim)permValues[i]);
        }
        transposeTensor = createHost2DeviceTensor(origTensor->getBufferDataType(),
                                                  origTensor->getHostDataSize() / origTensor->getElementSizeInBytes(),
                                                  name);
    }
    else
    {
        permutationArray.reserve(tensorDim);
        for (int i = 0; i < tensorDim; i++)
        {
            HB_ASSERT(permValues[i] < tensorDim, "permutation contains value bigger than the tensor dim");
            nodeParams.permutation.push_back((TransposePermutationDim)permValues[i]);
            permutationArray.push_back(static_cast<TransposePermutationDim>(finalPermValues[i]));
        }

        transposeTensor = getTensorAfterTranspose(*origTensor, permutationArray, name);
        transposeTensor->setPerChannelQuant(origTensor->isPerChannelQuant(), true);
        if (origTensor->isDataTypeMatchData())
        {
            transposeTensor->setAsDataTypeMatchData();
        }
    }

    if (isOutput)
    {
        nodeParams.input  = transposeTensor;
        nodeParams.output = origTensor;
    }
    else
    {
        nodeParams.input  = origTensor;
        nodeParams.output = transposeTensor;
    }
    m_transposesToInsert.push_back(std::move(nodeParams));
    m_origTensorToNewTensor.push_back(std::make_pair(origTensor, transposeTensor));
}

NodePtr TransposeInserter::createTransposeNodeFromParams(const TransposeNodeParams& params)
{
    auto originalTensorMapping =
        find_if(m_origTensorToNewTensor.begin(), m_origTensorToNewTensor.end(), [&params](const auto& tensorPair) {
            return tensorPair.first == params.input || tensorPair.second == params.input;
        });
    HB_ASSERT(originalTensorMapping != m_origTensorToNewTensor.end(),
              "mssing mapping from original to transposed tensor");
    const TensorPtr& origTensor = originalTensorMapping->first;
    const char* guid = origTensor->isShapeTensor()       ? NodeFactory::transposedShapeNodeTypeName :
                       origTensor->isHost2DeviceTensor() ? NodeFactory::transposeSliceH2DNodeTypeName :
                                                           NodeFactory::transposeNodeTypeName;
    synTransposeParams userParams = {};
    userParams.tensorDim          = origTensor->getDim();
    // always copy all the array
    memcpy(userParams.permutation, params.permutation.data(), sizeof(TransposePermutationDim) * MAX_DIMENSIONS_NUM);
    NodePtr transpose =
        NodeFactory::createInternalNode({params.input}, {params.output}, &userParams, guid, *params.nodeName);
    if (origTensor->isHost2DeviceTensor())
    {
        // fill transposed H2D tensor data
        transpose->inferOutputsSizes(synDeviceTypeInvalid, true);
        transpose->inferOutputsSizes(synDeviceTypeInvalid, false);
    }
    if (m_updateAnnotations)
    {
        NodeAnnotation &ann = transpose->getNodeAnnotation();
        ann.insertedNode = true;
    }
    return transpose;
}
