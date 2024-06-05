#include <habana_nodes/node_factory.h>
#include "bundlizer.h"
#include "reshape_aligner.h"
#include "habana_graph.h"
#include "slicing_utils.h"

static const std::string s_reshapeSuffix = "_reshape";
static const std::string s_reshapePrefix = "reshape_";
bool ReshapeAligner::alignProducerReshape(pBundleExpansion& expansionCandidate)
{
    HB_ASSERT(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT.value(),
              "Unexpected attempt to align producer reshapes despite globally disabled. Candidate to align: {} with "
              "reshape node: {}",
              expansionCandidate->nodeToStitch->getNodeName(),
              expansionCandidate->reshapeNode->getNodeName());

    LOG_DEBUG(SRAM_SLICE, "{}: Handling reshapes.", HLLOG_FUNC);

    /* Pre conditions validation*/
    if (!canAlignReshapes(expansionCandidate))
    {
        return false;
    }

    pNode&   tpcNode     = expansionCandidate->nodeToStitch;
    pTensor& mmeInput    = expansionCandidate->stitchedOperand->originalTensor;
    pNode&   reshapeNode = expansionCandidate->reshapeNode;

    /* Set TPC output to be [t3] - MME input                  */
    /* --[t1]-->(TPC) [t2]-->(Reshape)--[t3]-->(MME)--[t4]--> */
    /*            |-------------------->[t3]                  */
    /* Deduce the index of the output to be replaced [t2]     */
    unsigned int outputToBeReplaced = 0;
    bool         retVal             = deduceOutputIndex(outputToBeReplaced, tpcNode, reshapeNode);
    CHECK_RET_FALSE(retVal, "{}: didn't found output to be replaced!", HLLOG_FUNC);

    LOG_TRACE(GC,
              "{}: Replacing {} output. {} to {}",
              HLLOG_FUNC,
              tpcNode->getNodeName(),
              tpcNode->getOutput(outputToBeReplaced)->getName(),
              mmeInput->getName());

    GraphEditor::removeNode(m_graph, tpcNode);
    GraphEditor::removeNode(m_graph, reshapeNode);
    tpcNode->replaceOutput(outputToBeReplaced, mmeInput);

    /* Handle the rest of the tpcNode outputs.
     * Example: BN2 bf16 has two 4D nodes.
     * The first one handled above, the second will be handled here*/
    adjustTpcNodeOutputsShape(tpcNode, reshapeNode, mmeInput);

    /* Reshape TPC node input to the same shape as the next MME input - t3         */
    /* --[t1]-->(Reshape)--[t1']-->(TPC)                    [t3]-->(MME)--[t4]-->  */
    /*                               |--------------------->[t3]                   */
    /* At this point we know that the tpc node is elementwise                      */
    /* Find reshape on tpc node output - reshape has a single output               */
    insertReshapesBeforeTheBundle(tpcNode, reshapeNode);
    GraphEditor::addNode(m_graph, tpcNode);

    reverseReshapeIfRequired(reshapeNode, mmeInput);

    return true;
}

/* Pre-conditions validation:
 * Handle the following pattern: --[t1]-->(TPC)--[t2]-->(Reshape)--[t3]-->(MME)--[t4]-->*/
bool ReshapeAligner::canAlignReshapes(const pBundleExpansion& expansionCandidate) const
{
    pNode& tpcNode = expansionCandidate->nodeToStitch;
    pNode& reshape = expansionCandidate->reshapeNode;

    CHECK_RET_FALSE(HabanaGraph::runsOnTPC(tpcNode), "handleReshape: unsupported pattern, first node is not TPC!");
    CHECK_RET_FALSE(reshape->getNodeType() == Node::TYPE_INTERNAL_RESHAPE, "handleReshape: unsupported pattern!");
    TPCNodePtr pTpcNode = std::dynamic_pointer_cast<TPCNode>(tpcNode);
    CHECK_RET_FALSE(pTpcNode != nullptr, "handleReshape: node is not a TPC node");
    return true;
}

TensorVector promoteReshapeInputOperands(NodePtr const& originalReshapeNode, TensorPtr const& inputTensor)
{
    TensorVector inputOperands;
    inputOperands.reserve(2);
    inputOperands.push_back(inputTensor);

    if (inputTensor->isDynamicShape())
    {
        TensorPtr shapeTensor = originalReshapeNode->getInputs().back();
        HB_ASSERT((shapeTensor->getTensorType() == synTensorType::SHAPE_TENSOR),
                  "dynamic reshape node must have a shape tensor");
        inputOperands.push_back(shapeTensor);
    }
    return inputOperands;
}

NodePtr createReshapeNodeForInput(NodePtr const& originalReshapeNode, TensorPtr const& inputTensor)
{
    TensorPtr originalReshapeOutput = originalReshapeNode->getOutputs().front();
    TensorPtr reshapeOutput         = originalReshapeOutput->clone();
    reshapeOutput->resetAliasing();
    reshapeOutput->setName(inputTensor->getName() + s_reshapeSuffix);
    // we need to force the reshape output to match the correct data type
    // (matching the tpc node input) i.e. in case when the tpc node is a cast node
    // from one data type to another this is critical
    reshapeOutput->setElementType(inputTensor->getElementType());
    NodePtr      reshapeForInput;
    TensorVector inputOperands = promoteReshapeInputOperands(originalReshapeNode, inputTensor);

    reshapeForInput = NodeFactory::createNode(inputOperands,
                                              {reshapeOutput},
                                              nullptr,
                                              NodeFactory::reshapeNodeTypeName,
                                              s_reshapePrefix + inputTensor->getName());

    return reshapeForInput;
}

/* This function align the inputs of the tpc node to the shape of reshape output.
 * reshapeNode - reshape node from the bundle
 * tpcNode - tpc node from the bundle
 * (tpc)->(reshape)->(mme)*/
void ReshapeAligner::insertReshapesBeforeTheBundle(pNode& tpcNode, const pNode& reshapeNode) const
{
    /* Pre-condition: tpcNode is not in the graph */
    pTensor reshapedTensor = reshapeNode->getInputs().front();

    unsigned int tensorIndex = 0;

    /* Find inputs with same shape as reshape node input */
    for (const pTensor& inputTensor : tpcNode->getInputs())
    {
        if (inputTensor->getDim() == reshapedTensor->getDim())
        {
            if (inputTensor->getAllSizesInElements() == reshapedTensor->getAllSizesInElements())
            {
                /* Insert a reshape for current input tensor*/
                NodePtr reshapeForInput = createReshapeNodeForInput(reshapeNode, inputTensor);
                tpcNode->replaceInput(tensorIndex, reshapeForInput->getOutput(0));
                GraphEditor::addNode(m_graph, reshapeForInput);
            }
            else if (SlicedOperandUtils::isBroadcast(tpcNode))  // handle broadcast tpc node
            {
                /* Insert a reshape for current input tensor*/
                TensorPtr originalReshapeOutput = reshapeNode->getOutputs().front();

                // get reshape mapping from input to output
                const auto& reshapeMap =
                    SlicedOperandUtils::getReshapeOutputToInputMapping(originalReshapeOutput->getShape(), reshapedTensor->getShape());

                // set all initial dimensions same as original reshape output
                SizeArray reshapeDims = originalReshapeOutput->getAllSizesInElements();
                for (uint32_t dim = 0; dim < inputTensor->getDim(); dim++)
                {
                    // if dims are different, set 1 for all reshaped dims mapped to input dim
                    if (inputTensor->getAllSizesInElements()[dim] != reshapedTensor->getAllSizesInElements()[dim])
                    {
                        HB_ASSERT(inputTensor->getAllSizesInElements()[dim] == 1,
                                  "tpc node {} with broadcast where input {} dim is not 1",
                                  tpcNode->getNodeName(),
                                  inputTensor->getName());
                        for (uint32_t reshapedDim : reshapeMap[dim])
                        {
                            reshapeDims[reshapedDim] = 1;
                        }
                    }
                }
                TensorPtr reshapeOutput = std::make_shared<Tensor>(originalReshapeOutput->getDim(),
                                                                   reshapeDims.data(),
                                                                   originalReshapeOutput->getElementType());
                reshapeOutput->setName(inputTensor->getName() + s_reshapeSuffix);
                TensorVector inputOperands   = promoteReshapeInputOperands(reshapeNode, inputTensor);
                NodePtr      reshapeForInput = NodeFactory::createNode(inputOperands,
                                                                  {reshapeOutput},
                                                                  nullptr,
                                                                  NodeFactory::reshapeNodeTypeName,
                                                                  s_reshapePrefix + inputTensor->getName());

                tpcNode->replaceInput(tensorIndex, reshapeOutput);
                GraphEditor::addNode(m_graph, reshapeForInput);
            }
        }
        tensorIndex++;
    }
}

bool ReshapeAligner::deduceOutputIndex(unsigned int& outputIndexToBeReplaced, pNode tpcNode, pNode reshapeNode)
{
    outputIndexToBeReplaced = 0;

    for (const pTensor& tensor : tpcNode->getOutputs())
    {
        std::list<pNode> tensorConsumers = m_graph.getTensorConsumers(tensor);
        auto it = find(tensorConsumers.begin(), tensorConsumers.end(), reshapeNode);

        if (it != tensorConsumers.end())
        {
            return true;
        }
        outputIndexToBeReplaced++;
    }
    return false;
}

void ReshapeAligner::reverseReshapeIfRequired(pNode &reshapeNode, const pTensor &mmeInput) const
{
    pTensor reshapeInput = reshapeNode->getInputs()[0];

    /* If the input of the aligned reshape node is needed, the reshape should be reversed
     * 1.) The input has a consumer
     * 2.) The input is a persistent tensor*/
    if (m_graph.getNumberOfTensorConsumers(reshapeInput) != 0 || reshapeInput->isUserManagedDram())
    {
        /* Reverse the direction of the Reshapes nodes              */
        /* --[t1]-->(TPC)                    [t3]-->(MME)--[t4]-->  */
        /*            |--------------------->[t3]                   */
        /*                                   [t3]-->(Reshape)--[t2] */
        reshapeNode->replaceInput(0, mmeInput);
        reshapeNode->replaceOutput(0, reshapeInput);
        GraphEditor::addNode(m_graph, reshapeNode);
    }
}

void ReshapeAligner::adjustTpcNodeOutputsShape(const pNode &tpcNode, const pNode &reshapeNode, pTensor &mmeInput) const
{
    /* Pre-condition: tpcNode is not in the graph */
    pTensor      reshapeInput = reshapeNode->getInputs().front();
    unsigned int index        = 0;

    for (const pTensor& tensor : tpcNode->getOutputs())
    {
        if (tensor->getId() == mmeInput->getId())
        {
            index++;
            continue;
        }

        if (tensor->getAllSizesInElements() == reshapeInput->getAllSizesInElements())
        {
            TSize sizes[Tensor::c_tensorMaxDim];
            mmeInput->getAllSizesInElements(sizes, Tensor::c_tensorMaxDim);

            if (m_graph.getNumberOfTensorConsumers(tensor) != 0 || tensor->isUserManagedDram())
            {
                TensorPtr reshapeOut   = tensor;
                pTensor tpcNewOutput = tensor->clone();
                tpcNewOutput->resetAliasing();
                tpcNewOutput->setName(fmt::format("{}_reshape", tensor->getName()));
                tpcNewOutput->reshape(mmeInput->getDim(), sizes, nullptr);
                tpcNode->replaceOutput(index, tpcNewOutput);
                pNode reshapeForInput = NodeFactory::createNode({tpcNewOutput},
                                                                {reshapeOut},
                                                                nullptr,
                                                                NodeFactory::reshapeNodeTypeName,
                                                                fmt::format("reshape_{}", tensor->getName()));
                GraphEditor::addNode(m_graph, reshapeForInput);
            }
            else
            {
                tensor->reshape(mmeInput->getDim(), sizes, nullptr);
            }
        }
        index++;
    }
}

bool ReshapeAligner::alignConsumerReshape(pBundleExpansion &expansionCandidate)
{
    // Handles the pattern: (bundled producer) -> [stitched-operand] -> (reshape) -> [ ] -> (node-to-stitch)
    // If possible, the node-to-stitch will consume the stitched operand directly and other inputs and outputs
    // of that node with the same shape as the replaced input will be reshaped to match stitched-operand.

    HB_ASSERT(GCFG_ENABLE_SLICER_RESHAPE_ALIGNMENT.value(),
              "Unexpected attempt to align consumer reshapes despite globally disabled. Candidate to align: {} with "
              "reshape node: {}",
              expansionCandidate->nodeToStitch->getNodeName(),
              expansionCandidate->reshapeNode->getNodeName());

    LOG_TRACE(SRAM_SLICE, "{}: Handling reshapes.", HLLOG_FUNC);

    /* Pre conditions validation*/
    if (!canAlignReshapes(expansionCandidate))
    {
        return false;
    }

    const pNode&   tpcNode        = expansionCandidate->nodeToStitch;
    const pTensor& reshapeOutput  = expansionCandidate->reshapeNode->getOutput(0);

    GraphEditor::removeNode(m_graph, tpcNode);

    tpcNode->replaceTensor(reshapeOutput, expansionCandidate->stitchedOperand->originalTensor);
    alignConsumerInputShapes(expansionCandidate);
    alignConsumerOutputShapes(expansionCandidate);

    GraphEditor::addNode(m_graph, tpcNode);

    if (!reshapeOutput->isUserManagedDram() && m_graph.getNumberOfTensorConsumers(reshapeOutput) == 0)
    {
        SLC_TRACE("{}: Reshape output {} is neither persistent nor consumed. Removing reshape node {}.",
                  HLLOG_FUNC,
                  reshapeOutput->getName(),
                  expansionCandidate->reshapeNode->getNodeName());
        GraphEditor::removeNode(m_graph, expansionCandidate->reshapeNode);
    }
    return true;
}

// TODO [SW-8766] - Some of the code below can probably be shared with the producer handler. Refactor this module.

void ReshapeAligner::alignConsumerInputShapes(const pBundleExpansion& expansionCandidate)
{
    pNode&         tpcNode        = expansionCandidate->nodeToStitch;
    pNode&         reshapeNode    = expansionCandidate->reshapeNode;
    const pTensor& reshapeOutput  = reshapeNode->getOutput(0);
    const pTensor& reshapeInput   = reshapeNode->getInput(0);
    const pTensor& stitchedTensor = expansionCandidate->stitchedOperand->originalTensor;

    TensorVector origInputs = tpcNode->getInputs();
    TensorSet    handledTensors;
    for (const TensorPtr& tpcInput : origInputs)
    {
        if (handledTensors.count(tpcInput) == 1) continue;
        handledTensors.insert(tpcInput);

        if (tpcInput == stitchedTensor) continue;
        bool sameSizesNeedsReshape = false, sameDimNeedsReshape = false;
        if ((tpcInput->getAllSizesInElements() == reshapeOutput->getAllSizesInElements()) && // same sizes
            (tpcInput->getAllSizesInElements() != stitchedTensor->getAllSizesInElements() ||
             tpcInput->getDim() != stitchedTensor->getDim())) // Dimension count mismatch should be settled too
        {
            // sizes wise, this input looks identical to the reshape output (=to the other input that was stitched)
            sameSizesNeedsReshape = true;
        }
        else if (tpcInput->getDim() == reshapeOutput->getDim() &&
                 tpcInput->getDim() != stitchedTensor->getDim() &&
                 SlicedOperandUtils::areDimsDegenerated(tpcInput,
                                                        std::min(reshapeInput->getDim(), reshapeOutput->getDim()),
                                                        std::max(reshapeInput->getDim(), reshapeOutput->getDim()) - 1))
        {
            bool degeneratedDimsReshaped = SlicedOperandUtils::areDegeneratedDimsReshaped(reshapeNode);
            bool isInputBroadcast = SlicedOperandUtils::isBroadcast(tpcNode) && !tpcInput->compareGeometry(*reshapeOutput);

            if(degeneratedDimsReshaped || isInputBroadcast)
            {
                // even though not identical in the actual sizes, this input is identical in dimension size to te reshape
                // output (=other stitched input), and all the reshape dimension change are degenerated dims
                // or the input is broadcast
                sameDimNeedsReshape = true;
            }
        }

        if (sameSizesNeedsReshape || sameDimNeedsReshape)
        {
            pTensor tpcNewInput = tpcInput->clone(false /*copyAddress*/, false /*copyData*/);
            tpcNewInput->resetAliasing();
            tpcNewInput->setName(fmt::format("{}_reshaped", tpcInput->getName()));
            if (sameSizesNeedsReshape)
            {
                tpcNewInput->reshape(reshapeInput->getDim(), reshapeInput->getAllSizesInElements().data(), nullptr);
            }
            else // sameDimNeedsReshape - change dimension only
            {
                HB_ASSERT(tpcNewInput->resizeDims(reshapeInput->getDim()), "ReshapeAligner: align consumer input -"
                                                                              " can't resize dims");
            }
            TensorVector reshapeInputs = {tpcInput};
            pNode inputReshape = NodeFactory::createNode(reshapeInputs,
                                                         {tpcNewInput},
                                                         nullptr,
                                                         NodeFactory::reshapeNodeTypeName,
                                                         s_reshapePrefix + tpcInput->getName());
            GraphEditor::addNode(m_graph, inputReshape);
            tpcNode->replaceTensor(tpcInput, tpcNewInput);
        }
    }
}

void ReshapeAligner::alignConsumerOutputShapes(const pBundleExpansion& expansionCandidate)
{
    pNode&         tpcNode        = expansionCandidate->nodeToStitch;
    pNode&         reshapeNode    = expansionCandidate->reshapeNode;
    const pTensor& reshapeOutput  = reshapeNode->getOutput(0);
    const pTensor& reshapeInput   = reshapeNode->getInput(0);
    const pTensor& stitchedTensor = expansionCandidate->stitchedOperand->originalTensor;

    TensorVector origOutputs = tpcNode->getOutputs();
    for (const TensorPtr& tpcOutput : origOutputs)
    {
        bool sameSizesNeedsReshape = false, sameDimNeedsReshape = false;
        if ((tpcOutput->getAllSizesInElements() == reshapeOutput->getAllSizesInElements()) &&
            (tpcOutput->getAllSizesInElements() != stitchedTensor->getAllSizesInElements() ||
             tpcOutput->getDim() != stitchedTensor->getDim())) // Dimension count mismatch should be settled too
        {
            // sizes wise, this output looks identical to the reshape output
            sameSizesNeedsReshape = true;
        }
        else if (tpcOutput->getDim() == reshapeOutput->getDim() && tpcOutput->getDim() != stitchedTensor->getDim() &&
                 SlicedOperandUtils::areDegeneratedDimsReshaped(reshapeNode) &&
                 SlicedOperandUtils::areDimsDegenerated(tpcOutput,
                                                        std::min(reshapeInput->getDim(), reshapeOutput->getDim()),
                                                        std::max(reshapeInput->getDim(), reshapeOutput->getDim()) - 1))
        {
            // even though not identical in the actual sizes, this output is identical in dimension size to te reshape
            // output, and all the reshape dimension change are degenerated dims
            sameDimNeedsReshape = true;
        }
        if (sameSizesNeedsReshape || sameDimNeedsReshape)
        {
            pTensor tpcNewOutput = tpcOutput->clone();
            tpcNewOutput->resetAliasing();
            tpcNewOutput->setName(fmt::format("{}_reshaped", tpcOutput->getName()));
            if (sameSizesNeedsReshape)
            {
                tpcNewOutput->reshape(reshapeInput->getDim(), reshapeInput->getAllSizesInElements().data(), nullptr);
            }
            else // sameDimNeedsReshape - change dimension only
            {
                HB_ASSERT(tpcNewOutput->resizeDims(reshapeInput->getDim()), "ReshapeAligner: align consumer output -"
                                                                              " can't resize dims");
            }
            TensorVector reshapeInputs = {tpcNewOutput};
            pNode outputReshape = NodeFactory::createNode(reshapeInputs,
                                                          {tpcOutput},
                                                          nullptr,
                                                          NodeFactory::reshapeNodeTypeName,
                                                          s_reshapePrefix + tpcOutput->getName());
            GraphEditor::addNode(m_graph, outputReshape);
            tpcNode->replaceTensor(tpcOutput, tpcNewOutput);
        }
    }
}
