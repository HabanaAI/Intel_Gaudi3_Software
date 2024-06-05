
#include "habana_graph.h"
#include "passes.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "data_type_utils.h"

// TODO - Use Ori's function here -
// https://gerrit.habana-labs.com/#/c/293362/3/src/graph_compiler/passes/convert_buffers_to_float.cpp
// after it will merge
bool convertBufferDataToF32(TensorPtr& tensor)
{
    char*    data        = tensor->getData();
    unsigned numElements = tensor->getBufferSizeInBytes() / tensor->getElementSizeInBytes();
    switch (tensor->getBufferDataType())
    {
        case syn_type_float:
        case syn_type_int32:
        {
            // No need to convert
            break;
        }
        case syn_type_bf16:
        {
            LOG_TRACE(GC, "converting tensor {} buffer data type from bf16 to float32", tensor->getName());
            float* convertedData = bf16BufferTofloatBuffer((bf16_t*)data, numElements);
            tensor->setTensorBuffer(convertedData, numElements * sizeof(float), syn_type_float, true);
            delete convertedData;
            break;
        }
        case syn_type_fp16:
        {
            LOG_TRACE(GC, "converting tensor {} buffer data type from fp16 to float32", tensor->getName());
            float* convertedData = float16BufferToFloatBuffer((fp16_t*)data, numElements);
            tensor->setTensorBuffer(convertedData, numElements * sizeof(float), syn_type_float, true);
            delete convertedData;
            break;
        }
        default:
        {
            LOG_WARN(GC, "unsupported data type {} for tensor {}", tensor->getElementType(), tensor->getName());
            return false;
        }
    }

    return true;
}

static NodeSet getAllBNConvPatterns(HabanaGraph& g)
{
    Graph     pattern1;  // conv output is first input to batch norm
    bool      pattern1Status;
    {
        synConvolution3DParamsV2 params;

        pTensor convIFM = std::make_shared<Tensor>();
        pTensor convOFM = std::make_shared<Tensor>();
        pTensor convWGH = std::make_shared<Tensor>();
        pNode   convNode  = NodeFactory::createNode({convIFM, convWGH, nullptr, nullptr}, {convOFM}, &params, NodeFactory::convolutionNodeTypeName, "");
        pTensor BNGamma  = std::make_shared<Tensor>();
        pTensor BNBeta   = std::make_shared<Tensor>();
        pTensor BNOutput = std::make_shared<Tensor>();
        pTensor BNMean   = std::make_shared<Tensor>();
        pTensor BNVar = std::make_shared<Tensor>();
        pNode   BNNode   = NodeFactory::createGenericTPCNode({convOFM, BNGamma, BNBeta, BNMean, BNVar}, {BNOutput}, nullptr, "batch_norm_inf_f32", "");

        pattern1Status = pattern1.addNode(convNode);
        pattern1Status = pattern1Status && pattern1.addNode(BNNode);
    }

    if (pattern1Status)
    {
        // find all matches for above patterns
        return g.matchPatternWithSingleOutputNode(&pattern1, NodeTypeMatchingFunc);
    }
    else
    {
        LOG_DEBUG(GC, "Pattern build failed for BN-Conv fusing.");
    }
    return NodeSet();
}

static void fuseWeightTensor(float*                                 weightConvData,
                             const float*                           gammaBNData,
                             const std::shared_ptr<ConvolutionNode> convNode)
{
    LOG_TRACE(GC, "BNConv fusion - fusing the gamma tensor into the weight tensor");

    unsigned numOfFilters            = convNode->inputDimNameToSize(TENSOR_WEIGHT, 'K');
    unsigned weightOutputChannelsize = convNode->inputDimNameToSize(TENSOR_WEIGHT, 'C');
    unsigned rDimsize                = convNode->inputDimNameToSize(TENSOR_WEIGHT, 'R');
    unsigned sDimsize                = convNode->inputDimNameToSize(TENSOR_WEIGHT, 'S');

    unsigned totalWsizeWithoutK = weightOutputChannelsize * rDimsize * sDimsize;

    // Calculate the new weights
    for (int j = 0; j < numOfFilters; ++j)
    {
        unsigned index = j * totalWsizeWithoutK;
        for (int i = 0; i < totalWsizeWithoutK; ++i)
        {
            weightConvData[index + i] *= gammaBNData[j];
        }
    }

    // Unset dynamicRange for new W tensor in order to re-Calculate it next passes
    DynamicRange dynamicRange;
    TensorPtr weightConv = convNode->getInput(TENSOR_WEIGHT);

    dynamicRange.min   = 0;
    dynamicRange.max   = 1;
    dynamicRange.isSet = false;
    weightConv->setDynamicRange(dynamicRange);

}

static bool fuseBiasTensor(HabanaGraph&                           g,
                           const std::shared_ptr<ConvolutionNode> convNode,
                           unsigned                               numOfFilters,
                           TensorPtr&                             betaBN,
                           const std::string&                     convNodeName)
{
    float* betaBNData = reinterpret_cast<float*>(betaBN->getData());
    float* biasConvData = nullptr;
    synDataType biasElementType = syn_type_na;

    if (convNode->hasBias())
    {
        TensorPtr biasConv = convNode->getInput(TENSOR_BIAS);
        HB_ASSERT(numOfFilters == biasConv->getTotalElements(), "conv bias size must match the number of filters");
        if (!biasConv->isStaticParam())
        {
            LOG_ERR(GC, "In BNConv fusion, the conv bias must be static");
            return false;
        }
        // Convert bias data to float
        if (!convertBufferDataToF32(biasConv))
        {
            return false;
        }
        biasConvData = reinterpret_cast<float*>(biasConv->getData());
        biasElementType = biasConv->getElementType();
    }

    if (biasConvData != nullptr)
    {
        LOG_TRACE(GC, "BNConv fusion - fusing the beta tensor into the bias tensor");
        for (int i = 0; i < numOfFilters; ++i)
        {
            biasConvData[i] += betaBNData[i];
        }
    }
    else
    {
        LOG_TRACE(GC, "BNConv fusion - creating conv bias with the beta tensor data");

        float* newConvBias = new float[numOfFilters];

        for (int i = 0; i < numOfFilters; ++i)
        {
            newConvBias[i] = betaBNData[i];
        }

        if (biasElementType == syn_type_na)
        {
            // If there is no bias set new bias type according to beta BN type
            biasElementType = betaBN->getElementType();
        }

        TSize bias_sizes[Tensor::c_tensorMaxDim] = {1, 1, 1, 1, 1};
        bias_sizes[0] = numOfFilters;
        pTensor newBiasTensor                    = std::make_shared<Tensor>(1U, bias_sizes, biasElementType);
        newBiasTensor->bind(reinterpret_cast<char *>(newConvBias), true);
        newBiasTensor->setAsStaticParam(true);
        newBiasTensor->setAsBias();
        newBiasTensor->setName(convNodeName + "_bias");
        GraphEditor::editNode(g, convNode, [&](){
            convNode->addMMETensor(newBiasTensor, TENSOR_BIAS);
        });
    }
    return true;
}

static bool performBNConvFusion(HabanaGraph&                           g,
                                const std::shared_ptr<TPCNode>         batchNormNode,
                                const std::shared_ptr<ConvolutionNode> convNode,
                                const std::string&                     convNodeName)
{
    TensorPtr gammaBN    = batchNormNode->getInput(2);  // gamma is the third input of BN
    TensorPtr betaBN     = batchNormNode->getInput(1);  // beta is the second input of BN
    TensorPtr weightConv = convNode->getInput(TENSOR_WEIGHT);
    const std::optional<gc::Permutation>& permutedInputPerm = weightConv->getPermutation();
    if (permutedInputPerm && !permutedInputPerm.value().isIdentity())
    {
        LOG_WARN(GC, "Weight tensor {} is permuted, can't perform BNConv fusion", weightConv->getName());
        return false;
    }
    unsigned gammaBNDataSize = gammaBN->getTotalElements();
    unsigned betaBNDataSize = betaBN->getTotalElements();
    unsigned numOfFilters            = convNode->inputDimNameToSize(TENSOR_WEIGHT, 'K');

    if (numOfFilters != gammaBNDataSize || numOfFilters != betaBNDataSize )
    {
        LOG_WARN(GC,
                 "In BNConv fusion, the size of gamma and beta should match the number of filters (K) in the weights");
        LOG_TRACE(GC,
                  "Size of gamma {}, beta {} and number of filters (K) {}",
                  gammaBNDataSize,
                  betaBNDataSize,
                  numOfFilters);
        return false;
    }

    synDataType betaBufferDataType  = betaBN->getBufferDataType();
    synDataType wBufferDataType     = weightConv->getBufferDataType();
    synDataType gammaBufferDataType = gammaBN->getBufferDataType();

    if ((betaBufferDataType != gammaBufferDataType) || (betaBufferDataType != syn_type_single))
    {
        LOG_WARN(GC, "BNConv fusion support only F32 buffer data type for BN gamma and beta");
        LOG_WARN(GC,
                 "betaBufferDataType = {} gammaBufferDataType = {} wBufferDataType = {}",
                 betaBufferDataType,
                 gammaBufferDataType,
                 wBufferDataType);
        return false;
    }

    bool   isGammaStatic      = gammaBN->isStaticParam();
    bool   isBetaStatic       = betaBN->isStaticParam();
    bool   isWeightConvStatic = weightConv->isStaticParam();

    if (!isGammaStatic || !isBetaStatic || !isWeightConvStatic)
    {
        LOG_WARN(GC, "In BNConv fusion, the tensors for BN gamma/beta and Convolution weights must have static data");
        LOG_WARN(GC,
                 "isGammaStatic= {} isBetaStatic = {} isWeightConvStatic = {}",
                 isGammaStatic,
                 isBetaStatic,
                 isWeightConvStatic);
        return false;
    }

    // Convert weight data to float
    if (!convertBufferDataToF32(weightConv))
    {
        return false;
    }

    float* gammaBNData    = reinterpret_cast<float*>(gammaBN->getData());
    float* betaBNData     = reinterpret_cast<float*>(betaBN->getData());
    float* weightConvData = reinterpret_cast<float*>(weightConv->getData());

    if (gammaBNData == nullptr || betaBNData == nullptr || weightConvData == nullptr)
    {
        LOG_WARN(GC, "In BNConv fusion, the tensors for BN gamma/beta and Convolution weights must have not null");
        LOG_WARN(GC, "gammaBNData is null = {} betaBNData is null = {} weightConvData is null = {}",
                 gammaBNData == nullptr,
                 betaBNData == nullptr,
                 weightConvData == nullptr);
        return false;
    }

    if (!fuseBiasTensor(g, convNode, numOfFilters, betaBN, convNodeName))
    {
        return false;
    }
    fuseWeightTensor(weightConvData, gammaBNData, convNode);

    GraphEditor::removeNode(g, batchNormNode);
    GraphEditor::replaceOutput(g, convNode, TENSOR_OFM, batchNormNode->getOutput(0));
    return true;
}

bool fuseBNConv(HabanaGraph& g)
{
    if (!g.getInferenceMode())
    {
        LOG_DEBUG(GC,
                  "Conv + BN Fusion is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  __FUNCTION__);
        return true;
    }
    NodeSet matchingNodes  = getAllBNConvPatterns(g);

    for (pNode node : matchingNodes)
    {
        std::shared_ptr<TPCNode> batchNormNode = std::dynamic_pointer_cast<TPCNode>(node);
        HB_ASSERT(batchNormNode != nullptr && batchNormNode->isGuidPrefix("batch_norm"),
                  "Matched node for BNConv pattern is not BatchNorm");

        NodePtr batchNormNodeProducer = g.getTensorProducer(node->getInputs()[TENSOR_IFM]);
        HB_ASSERT(batchNormNodeProducer != nullptr, "Matched node for ConvBN pattern is a Null pointer");

        // if the producer (convolution) has consumers other than the BN, can't fuse
        if (!GraphEditor::canEliminateTensor(g, batchNormNodeProducer->getOutput(TENSOR_OFM)))
        {
            LOG_TRACE(GC,
                      "Can't fuse Conv {} with BN {} because the intermediate tensor can't be removed",
                      batchNormNodeProducer->getNodeName(),
                      batchNormNode->getNodeName());
            continue;
        }

        std::shared_ptr<ConvolutionNode> convNode = std::dynamic_pointer_cast<ConvolutionNode>(batchNormNodeProducer);
        HB_ASSERT(convNode != nullptr, "Matched node for BNConv pattern is not Convolution");
        if (convNode->is3DConvolution())
        {
            // TODO - Add support for 3D convolution
            LOG_TRACE(GC,
                      "Can't fuse Conv {} with BN {} because conv3D isn't supported for fusion",
                      batchNormNodeProducer->getNodeName(),
                      batchNormNode->getNodeName());
            continue;
        };

        //Since some of the passes still exist in Python package, there might be convolution that do relu/add cin
        if (convNode->getConvolutionParams().activation.reluEnable
            || convNode->getInput(TENSOR_CIN) != nullptr)
        {
            LOG_TRACE(GC,
                      "Can't fuse Conv {} with BN {} because the conv has relu enabled or Cin",
                      batchNormNodeProducer->getNodeName(),
                      batchNormNode->getNodeName());
            continue;
        }

        std::string convNodeName = batchNormNodeProducer->getNodeName();
        if (!performBNConvFusion(g, batchNormNode, convNode, convNodeName))
        {
            LOG_WARN(GC,
                     "Fusion conditions for nodes Conv {} with BN {} are not met, no fusion will be done",
                     batchNormNodeProducer->getNodeName(),
                     batchNormNode->getNodeName());
        }
    }

    return true;
}
