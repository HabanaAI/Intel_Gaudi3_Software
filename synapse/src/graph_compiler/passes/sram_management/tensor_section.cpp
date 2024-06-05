#include "graph_editor.h"
#include "node_factory.h"
#include "habana_graph.h"
#include "tensor_section.h"
#include "slicing_utils.h"
#include "habana_global_conf.h"
#include "tensor_view_node.h"
#include "tensor_view_shape_node.h"

TensorSection::TensorSection(const HalReader&                        halReader,
                             const Bundle::Solution::pSlicedOperand& slicedOperand,
                             uint32_t                                bundleIdx,
                             BundleType                              bundleType,
                             const Settable<uint64_t>&               multiBufferId,
                             uint64_t                                sectionIdx)
: m_halReader(halReader),
  m_origTensor(slicedOperand->originalTensor),
  m_tensorSlicer(slicedOperand, bundleIdx, multiBufferId),
  m_bundleIdx(bundleIdx),
  m_bundleType(bundleType),
  m_sectionIdx(sectionIdx)
{
}

pTensor TensorSection::addProduceSlice(const CoordArray& coord, uint32_t opIdx)
{
    // Create a clone of the original tensor, and resize it to the slice size
    pTensor slice = m_tensorSlicer.getSlice(m_halReader, coord, true /*force create*/);

    SliceInfo info(slice, opIdx);
    m_producers.insert(std::make_pair(coord, info));

    return slice;
}

pTensor TensorSection::addConsumeSlice(const CoordArray& coord, uint32_t opIdx)
{
    // If there's multiple producers, and a reduction tensor is not created yet
    bool multipleProducers = m_producers.count(coord) > 1 && m_reductionOutputs.find(coord) == m_reductionOutputs.end();

    // Create a clone of the original tensor, and resize it to the slice size, or get a cached slice if found
    pTensor slice = m_tensorSlicer.getSlice(m_halReader, coord, multipleProducers /*force create*/);

    SliceInfo info(slice, opIdx);
    std::set<SliceInfo>& slices = m_consumers[coord];
    auto insertionPoint = slices.insert(info);
    if (insertionPoint.second == false)
    {
        // Same slice is used again
        // If a DMA should get the slice, the DMA should run before the first operation
        // Thus take the earliest operation index
        if (opIdx < insertionPoint.first->m_opIdx)
        {
            slices.erase(insertionPoint.first);
            slices.insert(info);
        }
    }

    if (multipleProducers)
    {
        m_reductionOutputs.insert(std::make_pair(coord, info));
    }

    return slice;
}

void TensorSection::generateGraphSection(HabanaGraph& graph, bool shouldEvictTensor)
{
    addProducersReduction(graph);

    if (! m_concatProducers.empty())
    {
        handlePassThroughNodes(graph);
    }

    if(shouldEvictTensor)
    {
        if (! m_origTensor->inSram())
        {
            addProducersMemcpyNodes(graph);
        }
        addProducersConcat(graph);
        eliminateSingleProducer(graph);
    }
    // Handle consumers
    addConsumersMemcpyNodes(graph);
    addConsumersSplit(graph);
}

/*
    This function returns false if the inputTensor is ***for sure*** not a RMW output,
        returns true otherwise.
    Note: This function can return true on tensors that are not RMW operation outputs.
    Current condition: This function will return false if all the real producers of inputTensor are tpc nodes that do
   not require RMW on the reducted memory.
*/
bool TensorSection::canTensorBeRmwOutput(const HabanaGraph& graph, const TensorPtr& inputTensor)
{
    std::vector<std::pair<NodePtr, TensorPtr>> producersAndOutputs;
    if (auto inputTensorProducer = graph.getTensorProducer(inputTensor))
    {
        producersAndOutputs.emplace_back(inputTensorProducer, inputTensor);
    }

    auto deviceId = deviceTypeToDeviceID(graph.getDeviceType());

    while (!producersAndOutputs.empty())
    {
        auto [currProducer, currProducerOutput] = producersAndOutputs.back();
        producersAndOutputs.pop_back();
        if (currProducer->isLogicalOperation())  // Current producer is no a real producer, search for real producers
                                                 // based on its input tensors
        {
            for (const auto& producerInput : currProducer->getInputs())
            {
                if (producerInput != nullptr)
                {
                    if (auto parentProducer = graph.getTensorProducer(producerInput))
                    {
                        producersAndOutputs.emplace_back(parentProducer, producerInput);
                    }
                }
            }
        }
        else
        {
            /*
                Conditions on real producers should be placed here!
            */
            if (auto tpcNodeProducer = dynamic_cast<TPCNode*>(currProducer.get()))  // Current producer is a TPC node
            {
                auto outputIndex = tpcNodeProducer->getOutputIndexOfTensor(currProducerOutput);
                if (tpcNodeProducer->isOutputTensorRmw(outputIndex, deviceId))
                {
                    return true;
                }
            }
            else
            {
                return true;
            }
        }
    }
    return false;
}

NodePtr
TensorSection::addReduction(const HabanaGraph& graph, const TensorVector& inputs, const TensorVector& outputs) const
{
    static const std::string operation = "reduction";

    /*
        We would like to determine whether we seek an add reduction or a set reduction.
        If at least one reduction input is possibly a RMW output, we perform an add reduction.
        Else - we perform a set reduction.
        This logic is needed as a fix for ticket [SW-98982]
    */
    unsigned reductionOp = REDUCTION_UNORDERED_SET;
    for (const auto& reductionInput : inputs)
    {
        if (canTensorBeRmwOutput(graph, reductionInput))
        {
            reductionOp = REDUCTION_ADD;
            break;
        }
    }

    NodePtr reductionNode = NodeFactory::createNode(inputs,
                                                    outputs,
                                                    &reductionOp,
                                                    NodeFactory::reductionNodeTypeName,
                                                    generateNodeName(operation));

    return reductionNode;
}

void TensorSection::addProducersReduction(HabanaGraph& graph)
{
    static const std::string operation = "reduction";
    auto beginProd = m_producers.begin();
    while (beginProd != m_producers.end())
    {
        // Get the last producer with the same slice coordinate (endProd points just after the last)
        auto endProd = m_producers.upper_bound(beginProd->first);
        if (std::distance(beginProd, endProd) > 1)
        {
            // Handle multiple producers for the same coordinate
            // collect all inputs
            CoordArray coord = beginProd->first;
            TensorVector reductionInputs;
            for (auto inputIter = beginProd; inputIter != endProd; ++inputIter)
            {
                reductionInputs.push_back(inputIter->second.m_slice);
            }
            // check if the output tensor is a sliced tensor in this section
            CoordToSliceInfo::iterator reductionOutIter = m_reductionOutputs.find(coord);
            pTensor outputTensor;
            if (reductionOutIter != m_reductionOutputs.end())
            {
                // use the tensor created by the section
                outputTensor = reductionOutIter->second.m_slice;
            }
            else
            {
                // create a new tensor for the reduction output
                outputTensor = beginProd->second.m_slice->clone();
                outputTensor->setName(getTensorName(coord, operation));
                TensorSlicer::setSliceMemory(outputTensor, beginProd->second.m_slice->location() == TENSOR_IN_SRAM /*inSram*/);
            }

            NodePtr reductionNode = addReduction(graph, reductionInputs, {outputTensor});
            reductionNode->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, m_bundleType, beginProd->second.m_opIdx));

            LOG_DEBUG(BE_SLICER, "Create reduction node {}, on coordinate [{}], num of inputs {}",
                      reductionNode->getNodeName(),
                      toString(coord, ','),
                      reductionInputs.size());

            GraphEditor::addNode(graph, reductionNode);

            // Erase the handled tensors and insert new tensors needs to be handle
            SliceInfo newInfo(outputTensor, beginProd->second.m_opIdx);
            m_producers.erase(beginProd, endProd);
            m_concatProducers.insert(std::make_pair(coord, newInfo));
        }
        else if (beginProd->second.m_slice != m_origTensor)
        {
            m_concatProducers.insert(std::make_pair(beginProd->first, beginProd->second));
        }
        beginProd = endProd;
    };
}

void TensorSection::handlePassThroughNodes(HabanaGraph& graph)
{
    auto consumersIter = m_consumers.begin();
    while (consumersIter != m_consumers.end())
    {
        auto nextConsumeIter = consumersIter;
        ++nextConsumeIter;
        auto produceSliceIter = m_concatProducers.find(consumersIter->first);
        if (produceSliceIter != m_concatProducers.end())
        {
            bool passThrough = false;
            SliceInfo& producedSlice = produceSliceIter->second;

            auto consumeSliceIter = consumersIter->second.begin();
            while (consumeSliceIter != consumersIter->second.end())
            {
                if (consumeSliceIter->m_slice == producedSlice.m_slice)
                {
                    LOG_DEBUG(BE_SLICER, "Pass through detected for tensor {} on coordinate [{}]",
                              m_origTensor->getName(), toString(consumersIter->first, ','));
                    passThrough = true;
                    consumersIter->second.erase(consumeSliceIter);
                    break;
                }
                ++consumeSliceIter;
            }
            // Other slices should be moved to HBM and back when they are needed
            if (! consumersIter->second.empty())
            {
                bool addMemcpyToHbm = true;
                HB_ASSERT(passThrough, "Fail to pass through for SRAM tensor {}, coordinate: [{}] - execution schedule issue",
                      m_origTensor->getName(), toString(consumersIter->first, ','));
                if (!passThrough) // this is kind of fallback, and should happen only in release mode
                {
                    pNode producer = graph.getTensorProducer(producedSlice.m_slice);
                    if (producer->getNodeType() != Node::TYPE_INTERNAL_REDUCTION)
                    {
                        // Set the compute output directly to HBM
                        producedSlice.m_slice->setTensorInWorkspace();
                        addMemcpyToHbm = false;
                    }
                }

                pTensor hbmTensor = producedSlice.m_slice;
                if (addMemcpyToHbm)
                {
                    hbmTensor = addMemcpyNode(graph, produceSliceIter->first, producedSlice, nullptr, false /*hbmToSram*/);
                }

                for (const auto& consumeSlice : consumersIter->second)
                {
                    HB_ASSERT(producedSlice.m_opIdx < consumeSlice.m_opIdx, "consumer comes before producer");
                    addMemcpyNode(graph, produceSliceIter->first, consumeSlice, hbmTensor, true /*hbmToSram*/);
                }
                // Update the concat inputs with the hbm tensor so it won't be copied again
                producedSlice.m_slice = hbmTensor;
            }
            m_consumers.erase(consumersIter);
        }
        consumersIter = nextConsumeIter;
    }
}

void TensorSection::addProducersMemcpyNodes(HabanaGraph& graph)
{
    bool useOrigTensor = m_concatProducers.size() == 1;
    for (auto& sliceAndCoord : m_concatProducers)
    {
        auto& info = sliceAndCoord.second;
        const auto& coord = sliceAndCoord.first;
        pTensor hbmTensor = addMemcpyNode(graph, coord, info, useOrigTensor? m_origTensor : nullptr, false /*hbmToSram*/);

        // Update concat producers with the HBM tensor
        info.m_slice = hbmTensor;
    }
}

void TensorSection::addProducersConcat(HabanaGraph& graph)
{
    if (m_tensorSlicer.shouldUseTensorView())
    {
        addTensorViewNodeForSlices(graph, m_concatProducers, false /*realTensorIsInput*/);
    }
    else
    {
        addAggregateSlicesNodes(graph, m_concatProducers, true /*concat*/);
    }
}

void TensorSection::eliminateSingleProducer(HabanaGraph& graph)
{
    if (! m_concatProducers.empty())
    {
        pTensor tensorToReplace = m_concatProducers.begin()->second.m_slice;
        pNode producer = graph.getTensorProducer(tensorToReplace);
        GraphEditor::replaceTensor(graph, producer, tensorToReplace, m_origTensor);
        const auto& consumers = graph.getTensorConsumers(tensorToReplace);
        for (auto c : consumers)
        {
            GraphEditor::replaceTensor(graph, c, tensorToReplace, m_origTensor);
        }
        m_concatProducers.clear();
    }
}

void TensorSection::addAggregateSlicesNodes(HabanaGraph& graph, CoordToSliceInfo& coordToSlice, bool concat)
{
    uint32_t axis = 0;

    while (! coordToSlice.empty() && axis < SYN_MAX_TENSOR_DIM)
    {
        AxisBaseCoordToValue baseToAxisIdx;
        // Start from the inner dimension towards the outer one
        bool needAgg = fillAxisByBaseCoord(coordToSlice, axis, baseToAxisIdx);

        if (needAgg)
        {
            createAggSlicesNodesForAxis(graph, baseToAxisIdx, axis, coordToSlice, concat);
        }
        ++axis;
    }
    HB_ASSERT((coordToSlice.empty() || coordToSlice.size() == 1), "Failed to aggregate tensors");
}

void TensorSection::createAggSlicesNodesForAxis(HabanaGraph& graph,
                                                const AxisBaseCoordToValue& axisConcatCoords,
                                                uint32_t axis,
                                                CoordToSliceInfo& coordToSlice,
                                                bool concat)
{
    const std::string operation = concat ? NodeFactory::concatenateNodeLogicalInternalTypeName
                                         : (m_origTensor->isShapeTensor() ? NodeFactory::splitShapeNodeTypeName
                                                                          : NodeFactory::splitNodeInternalTypeName);
    for (const auto& axisConcatC : axisConcatCoords)
    {
        if (axisConcatC.second.size() <= 1) continue;
        TensorVector slices;
        // Get all concat inputs / split outputs
        uint32_t opIdx = gatherSlicesForAxisBaseCoord(axis, axisConcatC, coordToSlice, slices);

        pTensor aggTensor = m_origTensor;

        if (axisConcatCoords.size() > 1)
        {
            // There's some middle multi concats
            aggTensor = getAggregatedTensor(slices, axis);
            aggTensor->setName(getTensorName(axisConcatC.first, fmt::format("{}_axis_{}", operation, axis)));

            coordToSlice.insert(std::make_pair(axisConcatC.first, SliceInfo(aggTensor, opIdx)));
        }
        TensorVector aggTensorVec = {aggTensor};

        for (const auto& slice : slices)
        {
            slice->getTensorAnnotation().origBigTensor = m_origTensor;
        }

        pNode node = NodeFactory::createNode(concat ? slices : aggTensorVec,
                                             concat ? aggTensorVec : slices,
                                             &axis,
                                             operation,
                                             generateNodeName(operation));

        node->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, m_bundleType, opIdx));

        LOG_DEBUG(BE_SLICER, "Create {} node {}, opIdx {} on axis {}. Base coordinate [{}]",
                operation, node->getNodeName(), opIdx, axis, toString(axisConcatC.first, ','));

        GraphEditor::addNode(graph, node);
    }
}

void TensorSection::addTensorViewNodeForSlices(HabanaGraph& graph, CoordToSliceInfo& coordToSlice, bool realTensorIsInput)
{
    if (coordToSlice.size() <= 1)
    {
        // no slices to add, there's a single slice for this tensor
        return;
    }
    bool isShapeTensorView = m_origTensor->isShapeTensor();

    std::shared_ptr<TensorViewNode> tensorViewNode;

    if (isShapeTensorView)
    {
        tensorViewNode = std::make_shared<TensorViewShapeNode>(m_origTensor,
                                                               realTensorIsInput,
                                                               fmt::format("{}_shape_view", m_origTensor->getName()));
    }
    else
    {
        tensorViewNode = std::make_shared<TensorViewNode>(m_origTensor,
                                                          realTensorIsInput,
                                                          fmt::format("{}_tensor_view", m_origTensor->getName()));
    }

    uint32_t opIdx = std::numeric_limits<uint32_t>::max();
    for (const auto& coordAndSlice : coordToSlice)
    {
        pTensor slice = coordAndSlice.second.m_slice;
        SizeVector offsets = m_tensorSlicer.getSliceOffsets(coordAndSlice.first);

        tensorViewNode->addView(slice, offsets);  // Slice strides will be taken from full tensor

        // Update the op ID to be the minimal over all slices, to make sure DMA is scheduled before all of them
        opIdx = std::min(opIdx, coordAndSlice.second.m_opIdx);
        LOG_DEBUG(BE_SLICER,
                  "Add view for coordinate [{}], offsets [{}]",
                  toString(coordAndSlice.first, ','),
                  toString(offsets, ','));
    }
    tensorViewNode->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, m_bundleType, opIdx));

    GraphEditor::addNode(graph, tensorViewNode);
    coordToSlice.clear();
}

void TensorSection::addConsumersMemcpyNodes(HabanaGraph& graph)
{
    for (auto coordAndConsumers : m_consumers)
    {
        const CoordArray& coord = coordAndConsumers.first;
        const std::set<SliceInfo>& slices = coordAndConsumers.second;
        bool useOriginalTensor = slices.begin()->m_slice->getAllSizesInElements() == m_origTensor->getAllSizesInElements();
        if (slices.begin()->m_slice->location() == TENSOR_IN_SRAM && ! m_origTensor->inSram())
        {
            HB_ASSERT(!useOriginalTensor || slices.size() == 1, "should use original tensor when there is 1 slice");
            static const std::string operation = NodeFactory::memcpyNodeTypeName;
            pTensor hbmTensor = m_origTensor;
            if (! useOriginalTensor)
            {
                hbmTensor = slices.begin()->m_slice->clone();
                hbmTensor->setName(getTensorName(coord, operation));
                TensorSlicer::setSliceMemory(hbmTensor, false /*inSram*/);
            }
            uint32_t opIdx = std::numeric_limits<uint32_t>::max();

            for (const SliceInfo& s : slices)
            {
                opIdx = std::min(opIdx, s.m_opIdx);
                addMemcpyNode(graph, coord, s, hbmTensor, true /*hbmToSram*/);
            }
            if (! useOriginalTensor)
            {
                m_splitConsumers.insert(std::make_pair(coord, SliceInfo(hbmTensor, opIdx)));
            }
        }
        else if(useOriginalTensor)
        {
            for (const auto& tensorSlice : slices)
            {
                replaceConsumersTensor(graph, tensorSlice.m_slice);
            }
        }
        else
        {
            HB_ASSERT(slices.size() == 1, "Accessing same slice in original tensor (HBM) with different tensors");
            m_splitConsumers.insert(std::make_pair(coord, *slices.begin()));
        }
    }
}

void TensorSection::addConsumersSplit(HabanaGraph& graph)
{
    if (m_tensorSlicer.shouldUseTensorView())
    {
        addTensorViewNodeForSlices(graph, m_splitConsumers, true /*realTensorIsInput*/);
    }
    else
    {
        addAggregateSlicesNodes(graph, m_splitConsumers, false /*concat*/);
    }
}

void TensorSection::replaceConsumersTensor(HabanaGraph& graph, const pTensor& tensor)
{
    NodeList consumersList = graph.getTensorConsumers(tensor);
    for (auto consumer : consumersList)
    {
        GraphEditor::replaceTensor(graph, consumer, tensor, m_origTensor);
    }
}

bool TensorSection::fillAxisByBaseCoord(const CoordToSliceInfo& coordToSlice, uint32_t axis, AxisBaseCoordToValue& baseToAxisIdx)
{
    bool multiIdxForBaseCoord = false;
    // Collect axis indices by the base coordinate
    // Base coordinate is the coordinate with 0 at the relevant axis
    for (const auto& coordAndSlice : coordToSlice)
    {
        CoordArray baseCoord = coordAndSlice.first;
        baseCoord[axis] = 0; // turn the coordinate to base coordinate
        std::set<uint32_t>& axisIndices = baseToAxisIdx[baseCoord];
        axisIndices.insert(coordAndSlice.first[axis]);
        multiIdxForBaseCoord |= (axisIndices.size() > 1);
    }

    return multiIdxForBaseCoord;
}

uint32_t TensorSection::gatherSlicesForAxisBaseCoord(uint32_t axis,
                                                     const AxisBaseCoordAndValue& baseAndAxisIdx,
                                                     CoordToSliceInfo& coordToSlice,
                                                     TensorVector& outTensors)
{
    CoordArray coord = baseAndAxisIdx.first;
    uint32_t opIdx = std::numeric_limits<uint32_t>::max();
    // Prepare concat inputs
    for (uint32_t axisIdx : baseAndAxisIdx.second)
    {
        coord[axis] = axisIdx;
        auto infoIter = coordToSlice.find(coord);

        HB_ASSERT(infoIter != coordToSlice.end(), "Coordinate not found in self built container");

        if (infoIter != coordToSlice.end())
        {
            const SliceInfo& info = infoIter->second;
            opIdx = std::min(opIdx, info.m_opIdx);
            outTensors.push_back(info.m_slice);
            coordToSlice.erase(infoIter);
        }
    }

    return opIdx;
}

pTensor TensorSection::getAggregatedTensor(const TensorVector& slices, uint32_t axis)
{
    pTensor aggTensor = slices.front()->clone();
    SizeArray newShape = aggTensor->getAllSizesInElements();
    newShape[axis] = getAxisAggSize(slices, axis);
    aggTensor->reshape(aggTensor->getDim(), newShape.data(), nullptr);
    TensorSlicer::setSliceMemory(aggTensor, false /*inSram*/);

    return aggTensor;
}

uint32_t TensorSection::getAxisAggSize(const TensorVector& tensors, uint32_t axis)
{
    uint32_t ret = 0;
    for (const pTensor& t : tensors)
    {
        ret += t->getSizeInElements(axis);
    }

    return ret;
}

pTensor TensorSection::addMemcpyNode(HabanaGraph& graph, const CoordArray& coord, const SliceInfo& info, pTensor requestTensor, bool hbmToSram)
{
    static const std::string operation = NodeFactory::memcpyNodeTypeName;
    pTensor hbmTensor = requestTensor;

    if (info.m_slice->location() != TENSOR_IN_SRAM)
    {
        if ((info.m_slice->getElementType() == m_origTensor->getElementType()))
        {
            return info.m_slice;
        }
        HB_ASSERT(GCFG_ENABLE_PIPELINE_MANAGEMENT.value(), "reduction in HBM is only expected in pipeline management");
        // Not returning here in pipeline management to allow this function to add the memcopy node, knowing that it
        // will turn later to cast. This is a workaround for simple adding cast node after reduction, instead of adding
        // it explicitly if the output data type requires it.
    }

    if (hbmTensor == nullptr)
    {
        hbmTensor = info.m_slice->clone();
        hbmTensor->setName(getTensorName(coord, operation));
        hbmTensor->setElementType(m_origTensor->getElementType());
        TensorSlicer::setSliceMemory(hbmTensor, false /*inSram*/);
    }

    TensorVector sramTensorVec = {info.m_slice};
    TensorVector dramTensorVec = {hbmTensor};

    HB_ASSERT(!hbmTensor->inSram(), "Memcpy SRAM to SRAM");

    pNode memcpyNode = NodeFactory::createNode(hbmToSram ? dramTensorVec : sramTensorVec,
                                               hbmToSram ? sramTensorVec : dramTensorVec,
                                               nullptr,
                                               operation,
                                               generateNodeName(operation));

    memcpyNode->getNodeAnnotation().bundleInfo.set(BundleInfo(m_bundleIdx, m_bundleType, info.m_opIdx));

    LOG_DEBUG(BE_SLICER,
              "Add {} memcpy {}, opIdx {}, coordinate [{}]",
              hbmToSram ? "HBM to SRAM" : "SRAM to HBM",
              memcpyNode->getNodeName(),
              info.m_opIdx,
              toString(coord, ','));

    GraphEditor::addNode(graph, memcpyNode);

    return hbmTensor;
}

std::string TensorSection::getTensorName(const CoordArray& coord, const std::string& operation) const
{
    return fmt::format("{}_{}_{}", m_origTensor->getName(), toString(coord, '_'), operation);
}

std::string TensorSection::generateNodeName(const std::string& operation) const
{
    return fmt::format("{}_bundle_{}/{}/{}",
                       m_origTensor->getName(),
                       m_bundleIdx,
                       operation,
                       m_nameIdxGenerator[operation]++);
}

TensorSection::SliceInfo::SliceInfo(const pTensor& slice, uint32_t opIdx)
: m_slice(slice)
, m_opIdx(opIdx)
{
}

bool TensorSection::SliceInfo::operator<(const SliceInfo& rhs) const
{
    // Sort by the slice tensor
    return TensorComparator()(m_slice, rhs.m_slice);
}
