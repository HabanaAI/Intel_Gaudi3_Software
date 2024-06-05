
#include "habana_pass.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "cast_nodes_handler.h"
#include "data_type_utils.h"

/* Input swapping variation for binary nodes in the Gelu pattern */
typedef enum {
    SWAP_SCALAR1_MULT_INPUTS = 0,
    SWAP_ADD_X_INPUTS,
    SWAP_SCALAR2_MULT_INPUTS,
    SWAP_MULT_X_INPUTS,
    SWAP_PLUS1_INPUTS,
    MULT_HALF_FIRST,
    SWAP_MULT_HALF_INPUTS,
    NUM_INPUT_SWAPS_VARIATIONS /* Keep last */
}InputSwapsVariations;

/* Variations for the math term X^3 in the Gelu pattern */
typedef enum
{
    X_POW_3 = 0,
    X_POW2_MUL_X,
    X_MUL_X_POW2,
    X_MUL_X_MUL_X_VAR1,
    X_MUL_X_MUL_X_VAR2,
    NUM_X_IN_POW_3_VARIATIONS
}XinPowOf3Variations;

/* Assitance definitions for correct scalar values checking */
#define ACCEPTED_SCALAR_ERROR   0.01
#define SCALAR1_VALUE           0.044715
#define SCALAR1_LOWER_BOUNDARY (SCALAR1_VALUE - ACCEPTED_SCALAR_ERROR)
#define SCALAR1_UPPER_BOUNDARY (SCALAR1_VALUE + ACCEPTED_SCALAR_ERROR)
#define SCALAR2_VALUE           0.79788 // np.sqrt(2 / np.pi)
#define SCALAR2_LOWER_BOUNDARY (SCALAR2_VALUE - ACCEPTED_SCALAR_ERROR)
#define SCALAR2_UPPER_BOUNDARY (SCALAR2_VALUE + ACCEPTED_SCALAR_ERROR)

bool addMultXMultHalfToPattern(Graph* pattern, bool status, unsigned var, pTensor x, pTensor input)
{
    pTensor half         = std::make_shared<Tensor>();
    pTensor middleOutput = std::make_shared<Tensor>();
    pTensor output       = std::make_shared<Tensor>();

    pNode multHalf;
    pNode multX;
    if (var & (1 << MULT_HALF_FIRST))
    {
        if ((var & (1 << SWAP_MULT_HALF_INPUTS)) == 0)
        {
            multHalf = NodeFactory::createGenericTPCNode({input, half}, {middleOutput}, nullptr, "mult", "multHalf");
        }
        else
        {
            multHalf = NodeFactory::createGenericTPCNode({half, input}, {middleOutput}, nullptr, "mult", "multHalf");
        }

        if ((var & (1 << SWAP_MULT_X_INPUTS)) == 0)
        {
            multX = NodeFactory::createGenericTPCNode({x, middleOutput}, {output}, nullptr, "mult", "multX");
        }
        else
        {
            multX = NodeFactory::createGenericTPCNode({middleOutput, x}, {output}, nullptr, "mult", "multX");
        }
    }
    else
    {
        if ((var & (1 << SWAP_MULT_X_INPUTS)) == 0)
        {
            multX = NodeFactory::createGenericTPCNode({x, input}, {middleOutput}, nullptr, "mult", "multX");
        }
        else
        {
            multX = NodeFactory::createGenericTPCNode({input, x}, {middleOutput}, nullptr, "mult", "multX");
        }

        if ((var & (1 << SWAP_MULT_HALF_INPUTS)) == 0)
        {
            multHalf = NodeFactory::createGenericTPCNode({middleOutput, half}, {output}, nullptr, "mult", "multHalf");
        }
        else
        {
            multHalf = NodeFactory::createGenericTPCNode({half, middleOutput}, {output}, nullptr, "mult", "multHalf");
        }
    }

    status = status && pattern->addNode(multHalf);
    status = status && pattern->addNode(multX);

    return status;
}

bool handleMultByXMultByHalf(HabanaGraph& g, pNode lastNode, unsigned var, pNode& prevNode, NodeList& nodesToRemove)
{
    /* In this function the most external node will not be added to nodesToRemove,
     * and will be removed seperately as 'lastNode*/
    pNode multByXNode;
    pNode multByHalfNode;

    if (lastNode == nullptr)
    {
        LOG_DEBUG(GC, "Input node to MultByXMultByHalf pattern is NULL");
        return false;
    }

    if (var & (1 << MULT_HALF_FIRST))
    {
        multByXNode = lastNode;
        multByHalfNode = (var & (1 << SWAP_MULT_X_INPUTS)) ?
                         g.getTensorProducer(multByXNode->getInput(0)) : g.getTensorProducer(multByXNode->getInput(1));
        if(multByHalfNode == nullptr)
        {
            LOG_TRACE(GC, "mult by 0.5 node is NULL");
            return false;
        }

        prevNode = (var & (1 << SWAP_MULT_HALF_INPUTS)) == 0 ? g.getTensorProducer(multByHalfNode->getInput(0)) :
                   g.getTensorProducer(multByHalfNode->getInput(1));

        nodesToRemove.push_back(multByHalfNode);
    }
    else
    {
        multByHalfNode = lastNode;
        multByXNode = (var & (1 << SWAP_MULT_HALF_INPUTS)) == 0 ? g.getTensorProducer(multByHalfNode->getInput(0)) :
                      g.getTensorProducer(multByHalfNode->getInput(1));

        if(multByXNode == nullptr)
        {
            LOG_TRACE(GC, "mult by X node is NULL");
            return false;
        }
        prevNode = (var & (1 << SWAP_MULT_X_INPUTS)) ?
                   g.getTensorProducer(multByXNode->getInput(0)) : g.getTensorProducer(multByXNode->getInput(1));

        nodesToRemove.push_back(multByXNode);
    }

    int scalarInputIndex = (var & (1 << SWAP_MULT_HALF_INPUTS)) == 0 ? 1 : 0;
    float_t* pHalfData = (float_t*)multByHalfNode->getInput(scalarInputIndex)->getData();
    if (pHalfData == nullptr || *pHalfData != 0.5)
    {
        LOG_TRACE(GC, "Expected value of 0.5 on GELU mult node '{}' input({})",
                  multByHalfNode->getNodeName(), scalarInputIndex);
        return false;
    }

    return true;
}

bool addPlusXToPattern(Graph* pattern, bool status, unsigned var, pTensor x, pTensor input, pTensor output)
{
    pNode plusX;
    if ((var & (1 << SWAP_ADD_X_INPUTS)) == 0)
    {
        plusX = NodeFactory::createGenericTPCNode({x, input}, {output}, nullptr, "add", "plusX");
    }
    else
    {
        plusX = NodeFactory::createGenericTPCNode({input, x}, {output}, nullptr, "add", "plusX");
    }
    status = status && pattern->addNode(plusX);

    return status;
}

bool addXPow3ToPattern(Graph* pattern, bool status, unsigned pow3var, pTensor input, pTensor output)
{
    if (pow3var == X_POW_3)
    {
        pTensor three = std::make_shared<Tensor>();

        /* Adding x^3 operator to pattern */
        pNode pow3 = NodeFactory::createGenericTPCNode({input, three}, {output}, nullptr, "pow", "pow3");
        status = status & pattern->addNode(pow3);
    }
    else if (pow3var == X_POW2_MUL_X || pow3var == X_MUL_X_POW2)
    {
        pTensor xpow2 = std::make_shared<Tensor>();

        pNode pow2 = NodeFactory::createGenericTPCNode({input}, {xpow2}, nullptr, "pow2", "pow2");

        pNode multx;
        if (pow3var == X_POW2_MUL_X)
        {
            multx = NodeFactory::createGenericTPCNode({xpow2, input}, {output}, nullptr, "mult", "multxForPow");
        }
        else
        {
            multx = NodeFactory::createGenericTPCNode({input, xpow2}, {output}, nullptr, "mult", "multxForPow");
        }
        status = status & pattern->addNode(pow2);
        status = status & pattern->addNode(multx);
    }
    else if (pow3var == X_MUL_X_MUL_X_VAR1 || pow3var == X_MUL_X_MUL_X_VAR2)
    {
        pTensor xpow2 = std::make_shared<Tensor>();

        pNode pow2 = NodeFactory::createGenericTPCNode({input, input}, {xpow2}, nullptr, "mult", "xmulx");

        pNode xmultxpow2;
        if (pow3var == X_MUL_X_MUL_X_VAR1)
        {
            xmultxpow2 = NodeFactory::createGenericTPCNode({xpow2, input}, {output}, nullptr, "mult", "xmultxpow2");
        }
        else
        {
            xmultxpow2 = NodeFactory::createGenericTPCNode({input, xpow2}, {output}, nullptr, "mult", "xmultxpow2");
        }
        status = status & pattern->addNode(pow2);
        status = status & pattern->addNode(xmultxpow2);
    }
    else
    {
        LOG_WARN(GC, "expected X^3 variation {}, failing variation", pow3var);
        return false;
    }

    return status;
}

bool handleXPow3Pattern(HabanaGraph& g, pNode lastNode, unsigned pow3var, NodeList& nodesToRemove)
{
    if (pow3var == X_POW_3)
    {
        pNode pow3Node = lastNode;
        float_t* threeData = (float_t*)pow3Node->getInput(1)->getData();

        if (threeData == nullptr || *threeData != 3)
        {
            LOG_TRACE(GC, "Expected value of 3 on input(1) for GELU pow node {}", pow3Node->getNodeName());
            return false;
        }

        nodesToRemove.push_back(pow3Node);
    }
    else if (pow3var == X_POW2_MUL_X || pow3var == X_MUL_X_POW2 ||
             pow3var == X_MUL_X_MUL_X_VAR1 || pow3var == X_MUL_X_MUL_X_VAR2)
    {
        pNode xpow2multXNode = lastNode;
        pNode pow2Node = (pow3var == X_POW2_MUL_X || pow3var == X_MUL_X_MUL_X_VAR1) ?
                                g.getTensorProducer(xpow2multXNode->getInput(0))
                                                                 : g.getTensorProducer(xpow2multXNode->getInput(1));

        nodesToRemove.push_back(xpow2multXNode);
        nodesToRemove.push_back(pow2Node);
    }
    else
    {
        LOG_WARN(GC, "Unexpected pattern variation {}", pow3var);
        return false;
    }

    return true;
}

bool addMultScalarPattern(Graph* pattern, bool status, unsigned var, pTensor input, pTensor output, int scalarNum)
{
    pTensor scalar = std::make_shared<Tensor>();
    pNode multScalar;

    std::string nodeName = "multScalar" + std::to_string(scalarNum);

    /* Adding mult operator with 0.044715 to pattern */
    if ((scalarNum == 1 && (var & (1 << SWAP_SCALAR1_MULT_INPUTS)) != 0) ||
        (scalarNum == 2 && (var & (1 << SWAP_SCALAR2_MULT_INPUTS)) != 0))
    {
        multScalar = NodeFactory::createGenericTPCNode({scalar, input}, {output},
                                                        nullptr, "mult", nodeName);
    }
    else
    {
        multScalar = NodeFactory::createGenericTPCNode({input, scalar}, {output},
                                                        nullptr, "mult", nodeName);
    }
    status = status && pattern->addNode(multScalar);

    return status;
}

bool handleMultScalar(HabanaGraph& g, pNode lastNode, unsigned var, pNode& prevNode, NodeList& nodesToRemove, int scalarNum)
{
    float scalarExpectedVal = scalarNum == 1 ? SCALAR1_VALUE : SCALAR2_VALUE;
    float scalarLowerBound  = scalarNum == 1 ? SCALAR1_LOWER_BOUNDARY : SCALAR2_LOWER_BOUNDARY;
    float scalarUpperBound  = scalarNum == 1 ? SCALAR1_UPPER_BOUNDARY : SCALAR2_UPPER_BOUNDARY;

    bool swap = false; //scalar on input 1
    if ((scalarNum == 1 && (var & (1 << SWAP_SCALAR1_MULT_INPUTS))) ||
        (scalarNum == 2 && (var & (1 << SWAP_SCALAR2_MULT_INPUTS))))
    {
        swap = true;
    }

    pNode multNode = lastNode;

    float_t* scalar = swap ? (float_t*)multNode->getInput(0)->getData() :
            (float_t*)multNode->getInput(1)->getData();

    if (scalar == nullptr || *scalar < scalarLowerBound || *scalar >scalarUpperBound)
    {
        LOG_TRACE(GC, "Expected aprox. value of {} on GELU scalar{} mult node '{}'",
                  scalarNum, scalarExpectedVal, multNode->getNodeName());
        return false;
    }

    prevNode = swap ? g.getTensorProducer(multNode->getInput(1)) :
               g.getTensorProducer(multNode->getInput(0));

    nodesToRemove.push_back(multNode);

    return true;
}

void executeFuseGeluPass(HabanaGraph& g)
{
    auto   patternG = std::make_shared<Graph>();
    Graph* pattern  = patternG.get();

    int generatedGeluNodes = 0;

    /* In order to support combination of all modes, a bitmap is being used. where each mode enum specifies a
     * corresponding bit in the var iterable element */
    for (unsigned var = 0; var < (1 << NUM_INPUT_SWAPS_VARIATIONS); var++)
    {
        for (unsigned xPow3Var = 0; xPow3Var < NUM_X_IN_POW_3_VARIATIONS; xPow3Var++)
        {
            pattern->clear();
            NodeSet matchingNodes; // will hold matches for all variation of this pattern

            bool      patternStatus = true;

            /* Static tensors: */
            pTensor one = std::make_shared<Tensor>();

            /* Dynamic tensors: */
            pTensor x              = std::make_shared<Tensor>();
            pTensor xPow3          = std::make_shared<Tensor>();
            pTensor multScalar1Out = std::make_shared<Tensor>();
            pTensor plusXOut       = std::make_shared<Tensor>();
            pTensor multScalar2Out = std::make_shared<Tensor>();
            pTensor tanhOut        = std::make_shared<Tensor>();
            pTensor plus1Out       = std::make_shared<Tensor>();

            /* Adding x^3 operator to pattern */
            patternStatus = patternStatus && addXPow3ToPattern(pattern, patternStatus, xPow3Var, x, xPow3);

            /* Adding mult operator with 0.044715 to pattern */
            patternStatus = patternStatus && addMultScalarPattern(pattern, patternStatus, var, xPow3, multScalar1Out, 1);

            /* Adding 'Add X' operator to pattern */
            patternStatus =
                    patternStatus && addPlusXToPattern(pattern, patternStatus, var, x, multScalar1Out, plusXOut);

            /* Adding mult operator with np.sqrt(2 / np.pi) to pattern */
            patternStatus =
                    patternStatus && addMultScalarPattern(pattern, patternStatus, var, plusXOut, multScalar2Out, 2);

            /* Adding Tanh operator to pattern */
            pNode tanh = NodeFactory::createGenericTPCNode({multScalar2Out}, {tanhOut}, nullptr, "tanh", "tanh");
            patternStatus = patternStatus && pattern->addNode(tanh);

            /* Adding Plus1 operator to pattern */
            pNode plus1;
            if ((var & (1 << SWAP_PLUS1_INPUTS)) == 0)
            {
                plus1 = NodeFactory::createGenericTPCNode({tanhOut, one}, {plus1Out}, nullptr, "add", "plus1");
            }
            else
            {
                plus1 = NodeFactory::createGenericTPCNode({one, tanhOut}, {plus1Out}, nullptr, "add", "plus1");
            }
            patternStatus = patternStatus && pattern->addNode(plus1);

            /* Adding two following operators multHalf and MultX in different variations */
            patternStatus = patternStatus && addMultXMultHalfToPattern(pattern, patternStatus, var, x, plus1Out);

            if (patternStatus)
            {
                // find all matches for above patterns
                NodeSet matches = g.matchPatternWithSingleOutputNode(pattern, NodeTypeMatchingFunc);
                matchingNodes.insert(matches.begin(), matches.end());
            }
            else
            {
                LOG_DEBUG(GC, "Pattern-1, variation {} build failed for Gelu fusing.", var);
                continue;
            }

            for (pNode lastNode : matchingNodes)
            {
                NodeList nodesToRemove;

                pNode plus1Node = nullptr;

                if (handleMultByXMultByHalf(g, lastNode, var, plus1Node, nodesToRemove) == false)
                {
                    continue;
                }

                if (plus1Node == nullptr)
                {
                    LOG_TRACE(GC, "plus 1 node is NULL");
                    continue;
                }

                int oneInputIndex = (var & (1 << SWAP_PLUS1_INPUTS)) == 0 ? 1 : 0;
                float_t* oneData = (float_t*)plus1Node->getInput(oneInputIndex)->getData();
                if (oneData == nullptr || *oneData != 1)
                {
                    LOG_TRACE(GC, "Expected value of 1 on GELU add node '{}' input({})",
                              plus1Node->getNodeName(), oneInputIndex);
                    continue;
                }
                nodesToRemove.push_back(plus1Node);

                pNode tanhNode = g.getTensorProducer(plus1Node->getInput((oneInputIndex + 1) % 2));
                if (tanhNode == nullptr)
                {
                    LOG_TRACE(GC, "tanh node is NULL");
                    continue;
                }
                nodesToRemove.push_back(tanhNode);

                pNode multScalar2Node = g.getTensorProducer(tanhNode->getInput(0));
                if (multScalar2Node == nullptr)
                {
                    LOG_TRACE(GC, "mult scalar2 node is NULL");
                    continue;
                }

                pNode addXNode = nullptr;
                if (handleMultScalar(g, multScalar2Node, var, addXNode, nodesToRemove, 2) == false)
                {
                    continue;
                }

                if (addXNode == nullptr)
                {
                    LOG_TRACE(GC, "addX node is NULL");
                    continue;
                }

                nodesToRemove.push_back(addXNode);

                pNode multScalar1Node = (var & (1 << SWAP_ADD_X_INPUTS)) ?
                                        g.getTensorProducer(addXNode->getInput(0))
                                                                         : g.getTensorProducer(addXNode->getInput(1));
                if (multScalar1Node == nullptr)
                {
                    LOG_TRACE(GC, "mult scalar1 node is NULL");
                    continue;
                }

                pNode pow3Node = nullptr;
                if (handleMultScalar(g, multScalar1Node, var, pow3Node, nodesToRemove, 1) == false)
                {
                    continue;
                }

                if (pow3Node == nullptr)
                {
                    LOG_TRACE(GC, "pow3 node is NULL");
                    continue;
                }

                if (handleXPow3Pattern(g, pow3Node, xPow3Var, nodesToRemove) == false)
                {
                    continue;
                }


                // Need to check if all node that are about to be removed has only one output consumer
                bool status = true;
                for (pNode nodeToRemove : nodesToRemove)
                {
                    if (!GraphEditor::canEliminateTensor(g, nodeToRemove->getOutput(0)))
                    {
                        LOG_TRACE(GC, "node '{}' is not eligible for removal, can't fuse Gelu", nodeToRemove->getNodeName());
                        status = false;
                        break;
                    }
                }
                if (!status)
                {
                    continue;
                }

                /* Input and output for the fused node */
                pTensor geluInput  = pow3Node->getInput(0);
                pTensor geluOutput = lastNode->getOutput(0);

                /* The fused node */
                std::string geluNodeName = fmt::format("gelu_{}", generatedGeluNodes);
                generatedGeluNodes++;
                std::string_view guid   = "gelu";
                std::string_view suffix = getDtypeSuffixFromSynDataType(geluInput->getElementType());
                NodePtr          geluNode =
                    suffix.empty()
                                 ? NodeFactory::createGenericTPCNode({geluInput}, {geluOutput}, nullptr, guid, geluNodeName)
                                 : NodeFactory::createGenericTPCNode({geluInput},
                                                            {geluOutput},
                                                            nullptr,
                                                            fmt::format("{}_{}", guid, suffix),
                                                            geluNodeName);

                /* Removing all the pattern node, and adding the fused node */
                nodesToRemove.push_back(lastNode);
                auto ret = GraphEditor::replaceNodes(g, nodesToRemove, {geluNode});
                HB_ASSERT(ret == REPLACE_NODE_SUCCESS, "{}: failed to fuse gelu pattern nodes", __FUNCTION__);
                LOG_INFO(GC, "Gelu pattern was found and replaced by a single node '{}'", geluNodeName);
            }
        }
    }
}

bool fuseGelu(HabanaGraph& g)
{
    /* Trying to find the following pattern in the graph and replace ir with Gelu kernel
    Gelu(x) = x * 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))) */

    executeFuseGeluPass(g);

    return true;
}
