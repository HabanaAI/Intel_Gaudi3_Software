#include "batch_gemm_solvers.h"

#include <habana_nodes/habana_nodes.h>
#include <habana_nodes/node_factory.h>
#include <compilation_hal_reader.h>
#include "defs.h"
#include "spatial_slicing_solver.h"

bool BatchGemmSolver::effectiveForBundle()
{
    if (!isBundleBatchGemm(getBundle()))
    {
        return false;
    }
    // make sure the solver can actually slice the tensors to fit to SRAM
    createAllStrategies();
    return !getStrategies().empty();
}

static pBundle createBatchlessGemmBundle(const pNode& node)
{
    const pTensor& in0 = node->getInput(TENSOR_IFM);
    const pTensor& in1 = node->getInput(TENSOR_WEIGHT);
    const pTensor& out = node->getOutput(TENSOR_OFM);

    TSize in0TempDims[MAX_DIMENSIONS_NUM];
    TSize in1TempDims[MAX_DIMENSIONS_NUM];
    TSize outTempDims[MAX_DIMENSIONS_NUM];
    for (auto i = 0; i < MAX_DIMENSIONS_NUM; ++i)
    {
        in0TempDims[i] = i < DIM_GEMM_BATCH ? in0->getSizeInElements(i) : 1;
        in1TempDims[i] = i < DIM_GEMM_BATCH ? in1->getSizeInElements(i) : 1;
        outTempDims[i] = i < DIM_GEMM_BATCH ? out->getSizeInElements(i) : 1;
    }
    pTensor in0Temp {std::make_shared<Tensor>(in0->getDim(), in0TempDims, in0->getElementType())};
    pTensor in1Temp {std::make_shared<Tensor>(in1->getDim(), in1TempDims, in1->getElementType())};
    pTensor outTemp {std::make_shared<Tensor>(out->getDim(), outTempDims, out->getElementType())};

    const char* gemmType = nullptr;
    switch(node->getNodeType())
    {
    case Node::TYPE_BATCH_GEMM:         gemmType = "gemm";      break;
    case Node::TYPE_BATCH_GEMM_DEDX:    gemmType = "gemm_dedx"; break;
    case Node::TYPE_BATCH_GEMM_DEDW:    gemmType = "gemm_dedw"; break;
    default:
        HB_ASSERT(false, "createBatchlessGemmBundle called with non-batchgemm type");
    }

    const auto& tmpParams = static_cast<BatchGemmNode*>(node.get())->getGEMMParams();
    const auto  tmpNode   = NodeFactory::createNode({in0Temp, in1Temp}, {outTemp}, &tmpParams, gemmType, "tmpName");

    pBundle tmpBundle(new Bundle());
    tmpBundle->addNode(tmpNode);
    return tmpBundle;
}

static SlicingStrategyPtr pickGemmStrategy(const SlicingStrategyList& strategies)
{
    SlicingStrategyPtr res;
    for (const auto& candidateStrat : strategies)
    {
        if (!res)
        {
            res = candidateStrat;
            continue;
        }

        const auto& curM = res->getMetrics();
        const auto& newM = candidateStrat->getMetrics();

        const auto half = SlicingBrain::knobs.maxSRAMCapInBytes / 2;
        const bool curFitsHalf = curM.SRAMCapacity <= half;
        const bool newFitsHalf = newM.SRAMCapacity <= half;

        {
            const bool newPrefetchable = newFitsHalf || newM.isDoubleBuffered;
            const bool curPrefetchable = curFitsHalf || curM.isDoubleBuffered;
            if (newPrefetchable && !curPrefetchable)
            {
                res = candidateStrat;
                continue;
            }
        }

        if (curM.isDoubleBuffered == newM.isDoubleBuffered && curFitsHalf == newFitsHalf &&
            curM.SRAMCapacity < newM.SRAMCapacity)
        {
            res = candidateStrat;
            continue;
        }

        if (!newM.isDoubleBuffered && newFitsHalf)
        {
            res = candidateStrat;
            continue;
        }

        // Prefer the one larger than half if both double buffered (Current is smaller than half)
        if (curM.isDoubleBuffered && newM.isDoubleBuffered && !newFitsHalf)
        {
            res = candidateStrat;
            continue;
        }
    }
    return res;
}

void BatchGemmSolver::createAllStrategies()
{
    SLC_DEBUG("BatchGemm SRAM usage: applying on bundle {}", getBundle()->getName());
    if (!getStrategies().empty())
    {
        SLC_TRACE("BatchGemmSolver:{}: Strategies were already created by effectiveForBundle", HLLOG_FUNC);
        return;
    }

    // Solve a single GEMM by delegating to NonCD2DSolver
    auto          tmpBundle = createBatchlessGemmBundle(m_mmeNode);
    NonCD2DSolver solver(m_halReader, tmpBundle);
    if (!solver.effectiveForBundle())
    {
        SLC_DEBUG("BatchGemmSolver: can't use NonCD2DSolver because it is not effectiveForBundle");
        return;
    }

    solver.createAllStrategies();
    const auto& strategies = solver.getStrategies();

    // Pick a single GEMM strategy based on our knowladge of the bundle which
    // contains only the split GEMMs.
    // Both double buffered strategies and single buffered strategies with less
    // than 0.5 SRAM Capacity can provide prefetching.
    // Double buffering reuses address space which benefits other bundles
    // at the expanse of the current one. This leads to a preference of using
    // a single buffered strategy requiring <= 0.5 SRAM Capacity, if possible.
    // Among strategies with the same amount of buffers/ in the same half,
    // we prefer the larger slices (Which can potentially lead to less GEMMs).
    SlicingStrategyPtr s = pickGemmStrategy(strategies);
    if (!s)
    {
        SLC_ERR("BatchGemmSolver failed to find a strategy using NonCD2DSolver!");
        return;
    }

    // operands sizes might exceed the max sram capacity if changing single buffer to double buffer
    s->setAllowUpdateNumOfBuffers(false);

    // Adjust the single GEMM stategy to apply to the trivially batched case

    auto& data                            = s->getSlicingData();
    data.bundleTensors[0]->originalTensor = m_mmeNode->getInput(TENSOR_IFM);
    data.bundleTensors[1]->originalTensor = m_mmeNode->getInput(TENSOR_WEIGHT);
    data.masterOperand->originalTensor    = m_mmeNode->getOutput(TENSOR_OFM);

    for (auto i = (unsigned) DIM_GEMM_BATCH; i < data.masterOperand->originalTensor->getDim(); ++i)
    {
        data.traversalPattern.push_back(i);
        for (const auto& v : {data.bundleTensors[0], data.bundleTensors[1], data.masterOperand})
        {
            v->chunkDimensions[i] = 1;
            v->finalShape[i]      = v->originalTensor->getSizeInElements(i);
        }
    }

    data.setOutputSliceBackwardMapping(MMETriviallyBatchedSliceMapper::mapOutputToInputs(m_mmeNode,
                                                                                         data.bundleTensors[0],
                                                                                         data.bundleTensors[1],
                                                                                         data.masterOperand));

    addStrategy(s);
}

bool BatchTinyGemmSolver::effectiveForBundle()
{
    if (!GCFG_SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE.value())  return false;

    if (! isBundleBatchGemm(getBundle()))  return false;

    const auto& node = getBundle()->getNodes().front();

    // BGEMM DeDx/DeDw are not supported yet
    if (node->getNodeType() == Node::TYPE_BATCH_GEMM_DEDX ||
        node->getNodeType() == Node::TYPE_BATCH_GEMM_DEDW)
    {
        return false;
    }

    const auto* bgemm = dynamic_cast<BatchGemmNode*>(node.get());
    HB_ASSERT_PTR(bgemm);

    // Effective unless both operands are transposed
    if (bgemm->getGEMMParams().transpose_a &&
        bgemm->getGEMMParams().transpose_b)
    {
        return false;
    }

    // Effective only for symmetric bgemms
    if (!bgemm->isSymmetricLayout())
    {
        return false;
    }

    // Effective only if each GEMM has no more than 256 MME vectors in height or depth
    // TODO: This is HW limitation that should be solved in the MME stack
    if (bgemm->getOutput(0)->getSizeInBytes(0) > 256 * CompilationHalReader::getHalReader()->getMmeVectorSize() ||
        bgemm->getOutput(0)->getSizeInBytes(1) > 256 * CompilationHalReader::getHalReader()->getMmeVectorSize())
    {
        return false;
    }

    // Effective only if at least a single GEMM inputs fit SRAM
    uint64_t singleGemmInputSize = 0ull;
    for (const auto& input : bgemm->getInputs())
    {
        if (!input) continue;

        const auto& inputSizes = input->getAllSizesInElements();
        singleGemmInputSize += inputSizes[0] * inputSizes[1] * input->getElementSizeInBytes();
    }
    // singleGemm must fit to double buffer
    if (singleGemmInputSize > SlicingBrain::knobs.maxSRAMCapInBytes / 2)
    {
        return false;
    }

    return true;
}

static inline void
updateChunk(const std::array<pSlicedOperand, 3>& operands, const unsigned dim, const unsigned chunkSize)
{
    for (const pSlicedOperand& operand : operands)
    {
        operand->chunkDimensions[dim] = chunkSize;
    }
}

// return true if is chunk size on slicing dim bigger than 1
bool BatchTinyGemmSolver::calculateChunkSize(const pMmeSlicingStrategy&           strategy,
                                             const std::array<pSlicedOperand, 3>& operands,
                                             const unsigned                       dim,
                                             const unsigned                       maxChunkSize)
{
    unsigned sramOperandsSize = 0;
    for (const auto& operand : operands)
    {
        if (operand->resideInSRAM)
        {
            HB_ASSERT(operand->chunkDimensions[dim] == 1,
                      "chunk on dim {} is: {}, but should be 1",
                      dim,
                      operand->chunkDimensions[dim]);
            uint32_t numBuffers = SlicedOperandUtils::isTriviallySliced(operand) ? 1 : operand->numOfBuffers;
            sramOperandsSize += SlicedOperandUtils::getSliceSizeInBytes(operand) * numBuffers;
        }
    }
    unsigned chunkSize = std::min((unsigned) SlicingBrain::knobs.maxSRAMCapInBytes / sramOperandsSize, maxChunkSize);
    updateChunk(operands, dim, chunkSize);

    // optimize slice size to split more equaly between slices
    // example: batch of size 35 that fit to memory size, so chunk will be 16 and slices: [16, 16, 3]
    // but here we set chunk size to be ceil(35 / 3) = 12 so slices: [12, 12, 11]
    unsigned nofSlices     = SlicedOperandUtils::nofSlices(operands.at(2), dim);
    unsigned slicedDimSize = operands.at(2)->finalShape[dim];
    // this is ceil(roof) function
    unsigned optimalChunkSize = slicedDimSize / nofSlices + ((slicedDimSize % nofSlices != 0) ? 1 : 0);
    updateChunk(operands, dim, optimalChunkSize);

    return optimalChunkSize != 1;
}

void BatchTinyGemmSolver::fillChunksWithMaxPossibleSize(const pMmeSlicingStrategy& strategy,
                                                        const unsigned             maxBatchSize)
{
    StrategySlicingData&          data     = strategy->getSlicingData();
    std::array<pSlicedOperand, 3> operands = {data.bundleTensors[0], data.bundleTensors[1], data.masterOperand};

    // Start with a maximum GEMMs per slice
    // calculate chunks on batch dims that multiple them is less or equal to maxBatchSize var
    unsigned batchSize           = 1;
    bool     maxBatchSizeReached = false;
    for (unsigned dim = DIM_GEMM_BATCH; dim < data.masterOperand->originalTensor->getDim(); ++dim)
    {
        data.traversalPattern.push_back(dim);
        unsigned chunkSize;
        if (maxBatchSizeReached)
        {
            chunkSize = 1;
        }
        else if (batchSize * data.masterOperand->finalShape[dim] > maxBatchSize)
        {
            maxBatchSizeReached = true;
            chunkSize           = maxBatchSize / batchSize;
        }
        else
        {
            chunkSize = data.masterOperand->finalShape[dim];
            batchSize *= data.masterOperand->finalShape[dim];
        }
        updateChunk(operands, dim, chunkSize);
    }
}

void BatchTinyGemmSolver::createStrategySlices(const pMmeSlicingStrategy& strategy)
{
    StrategySlicingData&          data     = strategy->getSlicingData();
    std::array<pSlicedOperand, 3> operands = {data.bundleTensors[0], data.bundleTensors[1], data.masterOperand};

    // reduce GEMMs from each slice until less than half SRAM capacity is reached.
    for (unsigned dim = data.masterOperand->originalTensor->getDim() - 1; dim >= DIM_GEMM_BATCH; --dim)
    {
        unsigned maxChunkSize = data.masterOperand->chunkDimensions[dim];
        updateChunk(operands, dim, 1);
        // break if slice fit to SRAM and sliced dim chunk is bigger than 1
        if (strategy->recalculateSramCapacity().SRAMCapacity <= SlicingBrain::knobs.maxSRAMCapInBytes &&
            calculateChunkSize(strategy, operands, dim, maxChunkSize))
        {
            break;
        }
    }
}

void BatchTinyGemmSolver::createAllStrategies()
{
    SLC_DEBUG("BatchTinyGemm SRAM usage: applying on bundle {}", getBundle()->getName());
    // optional batch sizes that should be optimal, and also the max size that HW support
    constexpr std::array<unsigned, 3> potentialBatchSizes = {8, 12, 16};
    std::vector<unsigned>             batchSizes({GCFG_SRAM_SLICER_BATCHGEMM_MAX_BATCH_SIZE.value()});
    if (!GCFG_SRAM_SLICER_FORCE_BATCHGEMM_MAX_BATCH_SIZE.value())
    {
        for (unsigned batchSize : potentialBatchSizes)
        {
            if (batchSize < batchSizes.at(0))
            {
                batchSizes.push_back(batchSize);
            }
        }
    }
    // save the slices that already created to avoid duplicates
    std::set<SizeArray> slices;
    for (unsigned batchSize : batchSizes)
    {
        pMmeSlicingStrategy  strategy = MmeSlicingStrategy::createStrategyForMMENode(m_halReader, m_mmeNode);
        StrategySlicingData& data     = strategy->getSlicingData();
        strategy->setOutputIsInSRAM(false).setInputIsInSRAM(0, true).setInputIsInSRAM(1, true);
        fillChunksWithMaxPossibleSize(strategy, batchSize);
        createStrategySlices(strategy);
        // avoid duplicates
        if (slices.count(data.masterOperand->chunkDimensions) == 0)
        {
            slices.insert(data.masterOperand->chunkDimensions);
            data.setOutputSliceBackwardMapping(MMETriviallyBatchedSliceMapper::mapOutputToInputs(m_mmeNode,
                                                                                                 data.bundleTensors[0],
                                                                                                 data.bundleTensors[1],
                                                                                                 data.masterOperand));
            addStrategy(strategy);
        }
    }
}