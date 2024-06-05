#include "defs.h"
#include "compilation_hal_reader.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "bn_utils.h"
#include "tpc_kernel_names.h"

/*
 * create memset and reduction node
 */
BNUtils::ReductionRelatedNodesArr BNUtils::createNodesForSramReduction(unsigned           elements,
                                                                       unsigned           minElements,
                                                                       unsigned           numberOfDims,
                                                                       const std::string& baseName,
                                                                       bool               isBnBwd,
                                                                       bool               locateInSram)
{
    //For now we don't have a pass which sets SRAM to 0 before reduction. This might change in the future.
    HB_ASSERT(numberOfDims <= 2, "illegal batch norm flavour");
    const unsigned ELEMENTS_PER_CHANNEL = isBnBwd ? 2 : 3; // tpc kernel expects shape (C,2) in bwd
    ReductionRelatedNodesArr returnValues  = {nullptr};
    SizeArray                sigmaSizes           = {elements, ELEMENTS_PER_CHANNEL};
    SizeArray                minSigmaSizes        = {minElements, ELEMENTS_PER_CHANNEL};

    TensorPtr input = std::make_shared<Tensor>(numberOfDims,
                                               sigmaSizes.data(),
                                               syn_type_float,
                                               nullptr,
                                               nullptr,
                                               false,
                                               false,
                                               0,
                                               minSigmaSizes.data());

    input->setName(fmt::format("{}_in", baseName));

    TensorPtr zeros = input->clone(false, false);
    zeros->setName(fmt::format("{}_zeros", baseName));

    TensorPtr output = input->clone(false, false);
    output->setName(fmt::format("{}_out", baseName));

    if (locateInSram)
    {
        input->setTensorInSram();
        zeros->setTensorInSram();
        output->setTensorInSram();
    }

    NodePtr memsetNode                                                  = NodeFactory::createNode({},
                                                 {zeros},
                                                 nullptr,
                                                 NodeFactory::memsetNodeTypeName,
                                                 fmt::format("{}_memset_zero", baseName));
    NodePtr reductionNode                                               = NodeFactory::createNode({zeros, input},
                                                    {output},
                                                    nullptr,
                                                    NodeFactory::reductionNodeTypeName,
                                                    fmt::format("{}_reduction", baseName));
    input->getTensorAnnotation().tensorReductionInfo.isReductionEnabled = true;

    returnValues[eMemsetNode]    = memsetNode;
    returnValues[eReductionNode] = reductionNode;
    return returnValues;
}

/*
 * returns NodeList of bn_get_moments_st1 and st2 which can replace the input momentsNode
 */
NodeList BNUtils::getMoments(TensorPtr        inputFM,
                             TensorPtr&       outputMean,
                             TensorPtr&       outputSigma,
                             TensorPtr&       outputStd,
                             std::string_view baseName)
{
    NodeList retNodes;

    TSize channels = inputFM->getSizeInElements(0);
    TSize minChannels = inputFM->getMinimalSizeInElements(0);

    float    bhw      = (float)inputFM->getTotalElements() / (float)channels;

    // need to create output tensor in case outputMean is null
    if (outputMean == nullptr)
    {
        SizeArray meanSizes = {channels};
        outputMean = std::make_shared<Tensor>(1, meanSizes.data(), syn_type_float);
        outputMean->setName(fmt::format("{}out_mean", baseName));
    }

    /* Validate dType */
    HB_ASSERT(inputFM->getElementType() == syn_type_bf16 || inputFM->getElementType() == syn_type_float,
        "Inappropriate element type");

    /*
     * Stage 1: sum of the input feature map,
     * need to provide output tensor which is in SRAM and reduceable.
     */
    BNUtils::ReductionRelatedNodesArr sumNodes = BNUtils::createNodesForSramReduction(channels,
                                                                                      minChannels,
                                                                                      1,
                                                                                      fmt::format("{}_sum", baseName),
                                                                                      /*isBnBwd*/ false,
                                                                                      /*locateInSram*/ true);

    outputSigma = sumNodes[BNUtils::eReductionNode]->getOutput(0);

    NodePtr stage1Node = NodeFactory::createNode({inputFM},
                                                 {sumNodes[BNUtils::eReductionNode]->getInput(1)},
                                                 nullptr,
                                                 getMomentsGuid(1, FWD, inputFM->getElementType()),
                                                 fmt::format("{}_st1", baseName));

    retNodes.push_back(sumNodes[BNUtils::eMemsetNode]);
    retNodes.push_back(sumNodes[BNUtils::eReductionNode]);
    retNodes.push_back(stage1Node);

    /*
     * Stage 2: the output of the sum reduction node contains the sum over each channel. it will be the 2nd input of
     *          the st2 kernel
     *          the 2nd output should be reduceable as well
     */
    BNUtils::ReductionRelatedNodesArr sigmaSqNodes =
        BNUtils::createNodesForSramReduction(channels,
                                             minChannels,
                                             1,
                                             fmt::format("{}_sigma_squared", baseName),
                                             /*isBnBwd*/ false,
                                             /*locateInSram*/ true);
    NodePtr stage2Node = NodeFactory::createNode({inputFM, sumNodes[BNUtils::eReductionNode]->getOutput(0)},
                                                 {sigmaSqNodes[BNUtils::eReductionNode]->getInput(1), outputMean},
                                                 nullptr,
                                                 getMomentsGuid(2, FWD, inputFM->getElementType()),
                                                 fmt::format("{}_st2", baseName));

    ns_BatchnormGetMomentsKernel::Params st2Params = {1.0 / bhw};
    std::shared_ptr<TPCNode>             stage2TPC = std::dynamic_pointer_cast<TPCNode>(stage2Node);
    stage2TPC->storeParamsInBuffer(&st2Params, sizeof(ns_BatchnormGetMomentsKernel::Params));

    retNodes.push_back(sigmaSqNodes[BNUtils::eMemsetNode]);
    retNodes.push_back(sigmaSqNodes[BNUtils::eReductionNode]);
    retNodes.push_back(stage2Node);

    // setting outputStd as sigmaSqNodes[eReductionNode] output
    outputStd = sigmaSqNodes[BNUtils::eReductionNode]->getOutput(0);

    return retNodes;
}

bool BNUtils::createBn1Bn2NodesBwd(Bn1Bn2BwdInputs&  inputs,
                                   Bn1Bn2BwdOutputs& outputs,
                                   std::string       baseName,
                                   synDataType       dtype,
                                   bool              isTraining,
                                   NodeList&         nodesList,
                                   bool              locateInSram)
{
    //Batch norm BWD doesn't have params, so we will set the default params from FWD
    ns_BatchNormKernel::ParamsV2 fullBnParams {};
    fullBnParams.momentum = 0.1;
    fullBnParams.threshold.f = 1e-05;
    fullBnParams.epsilon = 1e-05;
    fullBnParams.isTraining = isTraining;

    if (inputs.mean->getSizeInElements(0) != inputs.istd->getSizeInElements(0))
    {
        LOG_ERR(GC, "splitBatchNorm- mismatch- mean size is {} istd size is {}",
                inputs.mean->getSizeInElements(0), inputs.istd->getSizeInElements(0));
        return false;
    }
    unsigned channels = inputs.mean->getSizeInElements(0);
    unsigned minChannels = inputs.mean->getMinimalSizeInElements(0);

    NodePtr resizeMean = BNUtils::createReshapeNode(inputs.mean, 2, BNUtils::eDirectionForward);
    NodePtr resizeIstd = BNUtils::createReshapeNode(inputs.istd, 2, BNUtils::eDirectionForward);
    NodePtr concatMeanIstd = BNUtils::createConcat1Dto2D(channels, minChannels, resizeMean->getOutput(0), resizeIstd->getOutput(0));
    BNUtils::ReductionRelatedNodesArr sumDotpReduction = BNUtils::createNodesForSramReduction(channels,
                                                                                              minChannels,
                                                                                              2,
                                                                                              baseName,
                                                                                              /*isBnBwd*/ true,
                                                                                              locateInSram);

    ns_BatchNormStage2Kernel::ParamsV2 stage2Params {};
    stage2Params.N                       = inputs.xIn->getTotalElements() / inputs.xIn->getSizeInElements(DIM_C);
    stage2Params.momentum                = fullBnParams.momentum;
    stage2Params.disable_runnings_update = 0;
    stage2Params.epsilon                 = fullBnParams.epsilon;
    stage2Params.isTraining              = isTraining;

    TensorVector bn2Outputs = {sumDotpReduction[BNUtils::eReductionNode]->getInput(1)};
    if (outputs.dZ != nullptr)
    {
        bn2Outputs.push_back(outputs.dZ);
    }
    std::string stageTwoName = fmt::format("{}_stage2", baseName);
    NodePtr     stage2Node =
        NodeFactory::createNode({inputs.xIn, concatMeanIstd->getOutput(0), inputs.gradIn, inputs.gradIn},
                                bn2Outputs,
                                nullptr,
                                getBN2Guid(BN_OPS_BN, BWD, dtype),
                                stageTwoName);
    std::shared_ptr<TPCNode> stage2TPC = std::dynamic_pointer_cast<TPCNode>(stage2Node);
    stage2TPC->storeParamsInBuffer(&stage2Params, sizeof(stage2Params));

    //This tensor is a placeholder and will have no use.
    TensorPtr tensorBetaFake =
        BNUtils::createWorkspaceVectorsTensor(channels, channels, 1, fmt::format("{}_beta_fake", baseName));

    TensorPtr outBetaFakeTensor = tensorBetaFake->clone();
    outBetaFakeTensor->setName(fmt::format("{}_resized", outBetaFakeTensor->getName()));

    TensorPtr outBetaFakeShape = tensorBetaFake->cloneGeometry();
    outBetaFakeShape->setShapeTensor(INPUT_DESCRIBING_SHAPE_TENSOR);

    NodePtr resizeGammaNode = BNUtils::createReshapeNode(inputs.gamma, 2, BNUtils::eDirectionForward);

    NodePtr extractShapeNode = NodeFactory::createNode({resizeGammaNode->getOutput(TENSOR_OFM)},
                                                       {outBetaFakeShape},
                                                       nullptr,
                                                       NodeFactory::extractShapeNodeTypeName,
                                                       fmt::format("{}_extract_shape", baseName));

    auto sliceParams = SliceNode::getDefaultSliceParams(tensorBetaFake);

    NodePtr resizeBetaNode = NodeFactory::createNode({tensorBetaFake, outBetaFakeShape},
                                                     {outBetaFakeTensor},
                                                     &sliceParams,
                                                     NodeFactory::logicalSliceBwdNodeTypeName,
                                                     fmt::format("{}_beta_fake_slice", baseName));

    NodePtr concatBetaGamma = BNUtils::createConcat1Dto2D(channels, minChannels, resizeBetaNode->getOutput(0), resizeGammaNode->getOutput(0));

    ns_BatchNormStage1Kernel::ParamsV2 stage1Params {};
    stage1Params.disable_beta_gamma_update = 0;
    stage1Params.N                         = stage2Params.N;
    stage1Params.isTraining = isTraining;
    std::string stageOneName               = fmt::format("{}_stage1", baseName);

    TensorPtr tensorGradBetaGamma =
        BNUtils::createWorkspaceVectorsTensor(channels, minChannels, 2, fmt::format("{}_grad_beta_gamma", baseName));
    TensorPtr stage1GradIn = outputs.dZ ? outputs.dZ : inputs.gradIn;

    TensorVector bn1Inputs = {inputs.xIn, stage1GradIn, concatMeanIstd->getOutput(0),
                              sumDotpReduction[BNUtils::eReductionNode]->getOutput(0),
                              concatBetaGamma->getOutput(0)};
    NodePtr projectShape = nullptr;

    NodePtr                  stage1Node         = NodeFactory::createNode(bn1Inputs,
                                                 {outputs.dX, tensorGradBetaGamma},
                                                 nullptr,
                                                 getBN1Guid(BWD, dtype),
                                                 stageOneName);
    unsigned splitDim = 1;
    NodePtr                  splitGradBetaGamma = NodeFactory::createNode({tensorGradBetaGamma},
                                                         {outputs.dBeta, outputs.dGamma},
                                                         &splitDim,
                                                         NodeFactory::splitNodeInternalTypeName,
                                                         fmt::format("{}split_grad_beta_gamma", baseName));
    std::shared_ptr<TPCNode> stage1TPC = std::dynamic_pointer_cast<TPCNode>(stage1Node);
    stage1TPC->storeParamsInBuffer(&stage1Params, sizeof(stage1Params));
    if (projectShape != nullptr)
    {
        nodesList.push_back(projectShape);
    }
    nodesList.push_back(resizeMean);
    nodesList.push_back(resizeIstd);
    nodesList.push_back(concatMeanIstd);
    nodesList.push_back(sumDotpReduction[BNUtils::eMemsetNode]);
    nodesList.push_back(sumDotpReduction[BNUtils::eReductionNode]);
    nodesList.push_back(stage2Node);
    nodesList.push_back(resizeBetaNode);
    nodesList.push_back(extractShapeNode);
    nodesList.push_back(resizeGammaNode);
    nodesList.push_back(concatBetaGamma);
    nodesList.push_back(stage1Node);
    nodesList.push_back(splitGradBetaGamma);
    return true;
}

void BNUtils::increaseDim(const TensorPtr& t, unsigned dim, unsigned factor)
{
    NSizeArray maxSizes = t->getAllNSizesInElements();
    NSizeArray minSizes = t->getNMinimalSizesInElements();
    maxSizes[dim] *= factor;
    minSizes[dim] *= factor;
    t->reshape(t->getDim(), maxSizes.data(), nullptr, minSizes.data());
}

void BNUtils::reduceDim(const TensorPtr& t, unsigned dim, unsigned factor)
{
    NSizeArray maxSizes = t->getAllNSizesInElements();
    NSizeArray minSizes = t->getNMinimalSizesInElements();
    HB_ASSERT(maxSizes[dim] % factor == 0, "illegal packing factor!");
    HB_ASSERT(maxSizes[dim] == minSizes[dim], "cannot support dynamic packing dim!");
    maxSizes[dim] /= factor;
    minSizes[dim] /= factor;
    t->reshape(t->getDim(), maxSizes.data(), nullptr, minSizes.data());
}

// given [C,W,H,N] input return [factor*C,W,H,N] output tile
TensorPtr BNUtils::tileTensor(const TensorPtr& input, NodeList& nodesList, unsigned factor)
{
    if (!input) return input;
    constexpr unsigned      dim    = 0;
    synDataType             dtype  = input->getElementType();
    auto                    guid   = fmt::format("tile_fwd{}", type2Str(dtype));
    auto                    name   = fmt::format("{}_tile", input->getName());
    ns_TileKernel::ParamsV2 params = {1, 1, 1, 1, 1};
    params.repeat[dim]             = factor;
    TensorPtr output               = input->clone(false, false, false);
    increaseDim(output, dim, factor);

    nodesList.push_back(NodeFactory::createNode({input}, {output}, &params, guid, name));
    return output;
}

// given [factor*C,W,H,N] input return [C,W,H,N] output average
TensorPtr BNUtils::avgTensor(const TensorPtr& input, NodeList& nodesList, unsigned factor)
{
    if (!input) return input;
    constexpr unsigned dim        = 0;
    synDataType        dtype      = input->getElementType();
    auto               reduceGuid = fmt::format("mean_fwd{}", type2Str(dtype));
    auto               reduceName = fmt::format("{}_reduce", input->getName());
    const auto&        splitGuid  = NodeFactory::splitNodeInternalTypeName;
    auto               splitName  = fmt::format("{}_split", input->getName());

    TensorPtr output = input->clone(false, false, false);
    reduceDim(output, dim, factor);

    HB_ASSERT(factor <= MAX_TENSOR_NR, "{}: illegal factor {}", HLLOG_FUNC, factor);
    TensorVector splitOutputs;
    for (unsigned i = 0; i < factor; i++)
    {
        splitOutputs.push_back(output->clone(false, false, false));
    }
    nodesList.push_back(NodeFactory::createNode({input}, splitOutputs, &dim, splitGuid, splitName));
    nodesList.push_back(NodeFactory::createNode(splitOutputs, {output}, nullptr, reduceGuid, reduceName));
    return output;
}

// given [C,W,H,N] output return [factor*C,W,H,N] input before slice
TensorPtr BNUtils::sliceTensor(const TensorPtr& output, NodeList& nodesList, unsigned factor)
{
    if (!output) return output;
    constexpr unsigned dim  = 0;
    const auto&        guid = NodeFactory::sliceNodeTypeName;
    auto               name = fmt::format("{}_slice", output->getName());

    TensorPtr input = output->clone(false, false, false);
    increaseDim(input, dim, factor);
    auto params = SliceNode::getDefaultSliceParams(input);
    params.ends = output->getAllNSizesInElements();

    nodesList.push_back(NodeFactory::createNode({input}, {output}, &params, guid, name));
    return input;
}

// given [C,W,H,N] input return [factor*C,W//factor,H,N] output reshape OR
// given [factor*C,W//factor,H,N] output return [C,W,H,N] input reshape
TensorPtr
BNUtils::packTensor(const TensorPtr& tensor, NodeList& nodesList, unsigned factor, eReshapeNodeDirection direction)
{
    if (!tensor) return tensor;
    constexpr unsigned dim  = 0;
    auto               guid = NodeFactory::staticReshapeNodeTypeName;
    auto               name = fmt::format("{}_pack", tensor->getName());

    TensorPtr other = tensor->clone(false, false, false);
    TensorPtr in = tensor, out = other;
    reduceDim(other, dim + 1, factor);
    increaseDim(other, dim, factor);
    if (direction == eDirectionBackward)
    {
        std::swap(in, out);
    }
    nodesList.push_back(NodeFactory::createNode({in}, {out}, nullptr, guid, name));
    return other;
}

bool BNUtils::createBn1Bn2NodesFwd(Bn1Bn2FwdInputs&        inputs,
                                   Bn1Bn2FwdOutputs&       outputs,
                                   float                   momentum,
                                   float                   epsilon,
                                   std::string             baseName,
                                   synDataType             dtype,
                                   bool                    isTraining,
                                   NodeList&               nodesList,
                                   bool                    locateInSram,
                                   std::optional<unsigned> packingFactor)
{
    // Take the original node's tensors
    if (inputs.beta->getSizeInElements(0) != inputs.gamma->getSizeInElements(0))
    {
        LOG_ERR(GC, "splitBatchNorm- node {} mismatch- beta size is {} gamma size is {}",
                baseName, inputs.beta->getSizeInElements(0), inputs.gamma->getSizeInElements(0));
        return false;
    }
    if (inputs.xIn->getTotalElements() != outputs.xOut->getTotalElements())
    {
        LOG_ERR(GC, "splitBatchNorm- node {} mismatch- Xin size is {} Xout size is {}",
                baseName, inputs.xIn->getTotalElements(), outputs.xOut->getTotalElements());
        return false;
    }

    unsigned channels = inputs.beta->getSizeInElements(0);
    unsigned minChannels = inputs.beta->getMinimalSizeInElements(0);
    int      size        = inputs.xIn->getTotalElements() / inputs.xIn->getSizeInElements(DIM_C);
    // Resize Beta and Gamma for concat
    NodePtr betaResizedNode = createReshapeNode(inputs.beta, 2, eDirectionForward);
    NodePtr gammaResizedNode = createReshapeNode(inputs.gamma, 2, eDirectionForward);
    NodePtr weightsNodeConcat =
        createConcat1Dto2D(channels, minChannels, betaResizedNode->getOutput(0), gammaResizedNode->getOutput(0));

    if (inputs.runningMeanIn->getSizeInElements(0) != inputs.runningVarIn->getSizeInElements(0))
    {
        LOG_ERR(GC, "splitBatchNorm- mismatch- running mean size is {} running var size is {}",
                inputs.beta->getSizeInElements(0), inputs.gamma->getSizeInElements(0));
        return false;
    }

    // Resize running mean and var for concat
    unsigned runningMeanVarMaxSize    = inputs.runningMeanIn->getSizeInElements(0);
    unsigned runningMeanVarMinSize    = inputs.runningMeanIn->getMinimalSizeInElements(0);
    NodePtr  runningMeanResizeNode    = createReshapeNode(inputs.runningMeanIn, 2, eDirectionForward);
    NodePtr  runningVarResizeNode     = createReshapeNode(inputs.runningVarIn, 2, eDirectionForward);
    NodePtr  runningMeanVarNodeConcat = createConcat1Dto2D(runningMeanVarMaxSize,
                                                          runningMeanVarMinSize,
                                                          runningMeanResizeNode->getOutput(0),
                                                          runningVarResizeNode->getOutput(0));
    // Mean and var for BN stage 2
    TensorPtr sigmas = nullptr;

    // BN stage 1
    if (isTraining)
    {
        ns_BatchNormStage1Kernel::ParamsV2 stage1Params {};
        stage1Params.disable_beta_gamma_update = 0;
        stage1Params.N                         = size;
        stage1Params.isTraining                = true;
        std::string stageOneName               = fmt::format("{}_stage1", baseName);

        BNUtils::ReductionRelatedNodesArr sigmaNodes = BNUtils::createNodesForSramReduction(runningMeanVarMaxSize,
                                                                                            runningMeanVarMinSize,
                                                                                            2,
                                                                                            baseName,
                                                                                            /*isBnBwd*/ false,
                                                                                            locateInSram);
        NodePtr stage1Node = NodeFactory::createNode({inputs.xIn, inputs.runningMeanIn},
                                                     {sigmaNodes[BNUtils::eReductionNode]->getInput(1)},
                                                     nullptr,
                                                     getBN1Guid(FWD, dtype),
                                                     stageOneName);

        std::shared_ptr<TPCNode> stage1TPC = std::dynamic_pointer_cast<TPCNode>(stage1Node);
        stage1TPC->storeParamsInBuffer(&stage1Params, sizeof(stage1Params));

        sigmas = sigmaNodes[BNUtils::eReductionNode]->getOutput(0);

        nodesList.push_back(sigmaNodes[BNUtils::eMemsetNode]);
        nodesList.push_back(sigmaNodes[BNUtils::eReductionNode]);
        nodesList.push_back(stage1Node);
    }
    else
    {
        // Precomputed by user
        sigmas = runningMeanVarNodeConcat->getOutput(0);
    }

    // BN stage 2
    ns_BatchNormStage2Kernel::ParamsV2 stage2Params {};
    stage2Params.momentum                = momentum;
    stage2Params.epsilon                 = epsilon;
    stage2Params.disable_runnings_update = isTraining ? 0 : 1;
    stage2Params.N                       = size;
    stage2Params.isTraining              = isTraining;
    std::string stageTwoName             = fmt::format("{}_stage2", baseName);

    if (packingFactor.has_value())
    {
        sigmas = avgTensor(sigmas, nodesList, packingFactor.value());
        sigmas = tileTensor(sigmas, nodesList, packingFactor.value());
    }

    TensorVector st2Inputs = {inputs.xIn,
                              inputs.runningMeanIn,
                              sigmas,
                              weightsNodeConcat->getOutput(0),
                              runningMeanVarNodeConcat->getOutput(0)};

    TensorPtr meanAndStdOut        = createWorkspaceVectorsTensor(runningMeanVarMaxSize,
                                                           runningMeanVarMinSize,
                                                           2,
                                                           fmt::format("{}_mean_and_std", baseName));
    TensorPtr runningMeanAndVarOut = createWorkspaceVectorsTensor(runningMeanVarMaxSize,
                                                                  runningMeanVarMinSize,
                                                                  2,
                                                                  fmt::format("{}_mean_and_var", baseName));

    NodePtr stage2Node = NodeFactory::createNode(st2Inputs,
                                                 {outputs.xOut, runningMeanAndVarOut, meanAndStdOut},
                                                 nullptr,
                                                 getBN2Guid(BN_OPS_BN, FWD, dtype),
                                                 stageTwoName);

    std::shared_ptr<TPCNode> stage2TPC = std::dynamic_pointer_cast<TPCNode>(stage2Node);
    stage2TPC->storeParamsInBuffer(&stage2Params, sizeof(stage2Params));

    // Split computed mean and variance, unused in inference
    if (isTraining)
    {
        unsigned splitDim               = 1;
        NodePtr  splitRunningMeanAndVar = NodeFactory::createNode({runningMeanAndVarOut},
                                                                 {outputs.runningMeanOut, outputs.runningVarOut},
                                                                 &splitDim,
                                                                 NodeFactory::splitNodeInternalTypeName,
                                                                 fmt::format("{}split_bn_mean_var", baseName));

        // In case we receive a tensor of 4 dimensions like [N,1,1,1]
        // should be fixed after [SW-55206] is resolved
        NodePtr meanResizeNode = createReshapeNode(outputs.meanOut, 1, eDirectionBackward);
        NodePtr varResizeNode  = createReshapeNode(outputs.stdOut, 1, eDirectionBackward);

        NodePtr splitMeanAndStd = NodeFactory::createNode({meanAndStdOut},
                                                          {meanResizeNode->getInput(0), varResizeNode->getInput(0)},
                                                          &splitDim,
                                                          NodeFactory::splitNodeInternalTypeName,
                                                          fmt::format("{}split_bn_mean_and_std", baseName));

        nodesList.push_back(splitRunningMeanAndVar);
        nodesList.push_back(splitMeanAndStd);
        nodesList.push_back(meanResizeNode);
        nodesList.push_back(varResizeNode);
    }

    nodesList.push_back(runningMeanResizeNode);
    nodesList.push_back(runningVarResizeNode);
    nodesList.push_back(betaResizedNode);
    nodesList.push_back(gammaResizedNode);
    nodesList.push_back(weightsNodeConcat);
    nodesList.push_back(runningMeanVarNodeConcat);
    nodesList.push_back(stage2Node);

    return true;
}

NodePtr BNUtils::createReshapeNode(TensorPtr inTensor, unsigned dims, eReshapeNodeDirection direction)
{
    TensorPtr outTensor = inTensor->clone(false, false);
    outTensor->setName(fmt::format("{}_resized", inTensor->getName()));
    outTensor->resizeDims(dims);
    if (direction == eDirectionBackward)
    {
        std::swap(inTensor, outTensor);
    }
    if (outTensor->getDim() > inTensor->getDim())
    {
        unsigned expandDim = inTensor->getDim();
        return NodeFactory::createNode({inTensor}, {outTensor}, &expandDim, NodeFactory::expandDimsNodeTypeName, "");
    }
    else
    {
        return NodeFactory::createNode({inTensor}, {outTensor}, nullptr, NodeFactory::squeezeNodeTypeName, "");
    }

}

TensorPtr BNUtils::createWorkspaceVectorsTensor(unsigned elements, unsigned minElements ,unsigned vectors, const std::string& name)
{
    SizeArray outputSizes    = {elements, vectors};
    SizeArray outputMinSizes = {minElements, vectors};
    TensorPtr outTensor      = std::make_shared<Tensor>(2,
                                                   outputSizes.data(),
                                                   syn_type_float,
                                                   nullptr,
                                                   nullptr,
                                                   false,
                                                   false,
                                                   0,
                                                   outputMinSizes.data());
    outTensor->setTensorInWorkspace();
    outTensor->setName(name);
    return outTensor;
}

NodePtr BNUtils::createConcat1Dto2D(unsigned size, unsigned minsize,TensorPtr input1, TensorPtr input2)
{
    SizeArray outputSizes = {size, 2};
    SizeArray outputMinSizes = {minsize, 2};
    unsigned concatDim = 1;
    std::string concatName     = fmt::format("{}_concat_with_{}", input1->getName(), input2->getName());
    TensorPtr   out            = std::make_shared<Tensor>(2U,
                                             outputSizes.data(),
                                             input1->getElementType(),
                                             nullptr,
                                             nullptr,
                                             false,
                                             false,
                                             0,
                                             outputMinSizes.data());

    out->setName(fmt::format("{}_out", concatName));
    if (input1->inSram() || input2->inSram())
    {
        out->setTensorInSram();
    }
    else
    {
        out->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
    }
    return NodeFactory::createNode({input1, input2},
                                   {out},
                                   &concatDim,
                                   NodeFactory::concatenateNodeInternalTypeName,
                                   concatName);
}

static float tpcUtil(const NSizeArray& sizes, synDataType dtype)
{
    static constexpr unsigned LOOP_UNROLL_IDX   = 1;
    static constexpr unsigned TPC_UNROLL_FACTOR = 4;
    const auto                tpcVectorSize     = CompilationHalReader::getHalReader()->getTpcVectorSize();

    TSize numUnrollVectors = CEIL(sizes[LOOP_UNROLL_IDX], TPC_UNROLL_FACTOR);
    float unrollUtil       = static_cast<float>(sizes[LOOP_UNROLL_IDX]) / (numUnrollVectors * TPC_UNROLL_FACTOR);

    TSize fcdSize    = dataTypeSizeInBytes(dtype) * sizes[0];
    TSize numVectors = CEIL(fcdSize, tpcVectorSize);
    float vectorUtil = static_cast<float>(fcdSize) / (numVectors * tpcVectorSize);
    return unrollUtil * vectorUtil;
}

static bool isValidPackingFactor(const TensorPtr& ifm, unsigned packingFactor)
{
    return ifm->getSizeInElements(1) % packingFactor == 0;
}

static bool isCandidateForPacking(const TensorPtr& ifm, bool isTraining)
{
    static constexpr unsigned LOOP_UNROLL_IDX   = 1;
    static constexpr unsigned TPC_UNROLL_FACTOR = 4;
    if (!GCFG_ENABLE_BN_FCD_PACKING.value()) return false;
    if (!isTraining) return false;  // no usage of stage1
    if (ifm->getDim() < 2) return false;
    if (ifm->isDynamicDim(0) || ifm->isDynamicDim(1)) return false;  // cannot pack dynamic dim

    return true;
}

static float calcPackingFactorUtil(const TensorPtr& ifm, unsigned packingFactor)
{
    NSizeArray sizes = ifm->getAllNSizesInElements();
    HB_ASSERT(ifm->getDim() >= 2, "{}: illegal dim {}", HLLOG_FUNC, ifm->getDim());
    HB_ASSERT(sizes[1] % packingFactor == 0, "{}: illegal factor {}", HLLOG_FUNC, packingFactor);

    sizes[1] /= packingFactor;
    sizes[0] *= packingFactor;
    float util = tpcUtil(sizes, ifm->getElementType());
    LOG_TRACE(GC, "{}: size: {}, packing factor {} util  {}", HLLOG_FUNC, ifm->getDimSizesStr(), packingFactor, util);
    return util;
}

std::optional<unsigned> BNUtils::shouldOptimizeLowFCD(Bn1Bn2FwdInputs&  inputs,
                                                      Bn1Bn2FwdOutputs& outputs,
                                                      bool              isTraining,
                                                      std::string_view  baseName)
{
    static constexpr unsigned MIN_PACKING_FACTOR = 2;
    // since we will use a single "mean" kernel to reduces sigmas tensor,
    // this is the maximum number of inputs that we accept.
    static constexpr unsigned MAX_PACKING_FACTOR = MAX_TENSOR_NR;
    // penalty for packing (adding extra compute and syncs). This was found by experimenting and hardcoded here.
    static constexpr float PACKING_PENALTY = (float)3 / 4;
    const TensorPtr&       ifm             = inputs.xIn;
    if (!isCandidateForPacking(ifm, isTraining)) return std::nullopt;

    float baseUtilization = calcPackingFactorUtil(ifm, 1);
    if (baseUtilization >= PACKING_PENALTY) return std::nullopt;  // best utilization without packing.

    // find best packing factor with baseline being no packing
    std::optional<unsigned> factor          = std::nullopt;
    float                   bestUtilization = baseUtilization / PACKING_PENALTY;
    for (unsigned packingFactor = MIN_PACKING_FACTOR; packingFactor < MAX_PACKING_FACTOR; packingFactor++)
    {
        if (!isValidPackingFactor(ifm, packingFactor)) continue;
        float util = calcPackingFactorUtil(ifm, packingFactor);
        if (util > bestUtilization)
        {
            bestUtilization = util;
            factor          = packingFactor;
        }
    }
    if (factor.has_value())
    {
        LOG_DEBUG(GC,
                  "applying low FCD optimization on BN {}, with size: {}. base utilization: {}, new utilization:{}, "
                  "with factor {}",
                  baseName,
                  ifm->getDimSizesStr(),
                  baseUtilization,
                  bestUtilization,
                  factor.value());
    }
    return factor;
}

void BNUtils::optimizeLowFCD(unsigned          factor,
                             Bn1Bn2FwdInputs&  inputs,
                             Bn1Bn2FwdOutputs& outputs,
                             NodeList&         nodesList,
                             std::string_view  baseName)
{
    inputs.beta            = tileTensor(inputs.beta, nodesList, factor);
    inputs.gamma           = tileTensor(inputs.gamma, nodesList, factor);
    inputs.runningMeanIn   = tileTensor(inputs.runningMeanIn, nodesList, factor);
    inputs.runningVarIn    = tileTensor(inputs.runningVarIn, nodesList, factor);
    inputs.residualAddIn   = packTensor(inputs.residualAddIn, nodesList, factor, eDirectionForward);
    inputs.xIn             = packTensor(inputs.xIn, nodesList, factor, eDirectionForward);
    outputs.meanOut        = sliceTensor(outputs.meanOut, nodesList, factor);
    outputs.runningMeanOut = sliceTensor(outputs.runningMeanOut, nodesList, factor);
    outputs.runningVarOut  = sliceTensor(outputs.runningVarOut, nodesList, factor);
    outputs.stdOut         = sliceTensor(outputs.stdOut, nodesList, factor);
    outputs.xOut           = packTensor(outputs.xOut, nodesList, factor, eDirectionBackward);
}
