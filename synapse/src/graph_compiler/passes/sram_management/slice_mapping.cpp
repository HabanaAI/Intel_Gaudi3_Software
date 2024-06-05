#include "slice_mapping.h"
#include <memory>
#include <utility>
#include <vector>
#include "node.h"
#include "slicing_utils.h"
#include "slicing_brain.h"
#include "sram_management/bundle.h"

using namespace gc::access_pattern;

class MMEBackwardSliceMapping : public BackwardSliceMapping
{
public:
    MMEBackwardSliceMapping(pNode mmeNode, const std::vector<pSlicedOperand>& inOperands, pSlicedOperand opOut)
    : BackwardSliceMapping(inOperands, opOut)
    , m_mmeNode(std::move(mmeNode)), m_dimController(std::move(m_mmeNode))
    {
    }

    SliceReferenceList getInputs(const SliceRefCommonDimIdxPair& slice) const override;

    virtual pBackwardSliceMapping clone(std::vector<pSlicedOperand>& inputOperands,
                                        pSlicedOperand& outputOperand) override
    {
        auto shapeTensor = inputOperands.size() == 3 && inputOperands[2]->originalTensor->isShapeTensor() ?
                           inputOperands[2] : nullptr;

        HB_ASSERT(inputOperands.size() == 2 || shapeTensor, "should be 2 inputs or 3 with shape tensor");

        return pBackwardSliceMapping(new MMEBackwardSliceMapping(m_mmeNode, inputOperands,
                                                                 outputOperand));
    }

protected:
    enum class OperandType {opA, opB};
    // set coordinate values inside sliceRef for all relevant dims according to operand type
    // the non common dim values are filled according to relevant output part (height/width)
    virtual void fillNonCommonDim(OperandType opType, const pSliceReference& inSliceRef,
                                  const pSliceReference& outSliceRef) const;
    // set coordinate values inside sliceRef for all relevant dims according to operand type
    // the common dim values are filled according to given coord having values between 0 and num of total slices
    virtual void fillCommonDim(OperandType opType, const pSliceReference& sliceRef, unsigned coord) const;

    pNode m_mmeNode;
    const MmeDimController m_dimController;
};

class MMETriviallyBatchedBackwardSliceMapping : public MMEBackwardSliceMapping
{
public:

    MMETriviallyBatchedBackwardSliceMapping(pNode mmeNode, const std::vector<pSlicedOperand>& inOperands, pSlicedOperand opOut)
    : MMEBackwardSliceMapping(std::move(mmeNode), inOperands, std::move(opOut))
    {
    }

    SliceReferenceList getInputs(const SliceRefCommonDimIdxPair& slice) const override;

    pBackwardSliceMapping clone(std::vector<pSlicedOperand>& inputOperands,
                                        pSlicedOperand& outputOperand) override
    {
        HB_ASSERT(inputOperands.size() == 2, "should be only 2 inputs");
        return pBackwardSliceMapping(
            new MMETriviallyBatchedBackwardSliceMapping(m_mmeNode, inputOperands, outputOperand));
    }
};

// This method is responsible to return the mapped inputs (as SliceReferences) from the given slice
// the slice given is a pair of a SliceReference and a coordinate in the partial dimension.
// for each input {operandA, operandB} the coordinates will be filled
// in the following way -
// input common dim coordinates are the output partials coordinate
// input non-common dim are the output non-common dim
// examples : (p = partials dimension)
// mapping for TYPE_CONVOLUTION:
// Map {(b, h, w, k), p} -> {(b, h, w, p), (0, 0, p, k)}
//
// mapping for TYPE_DEDW:
// bhw are common but tensors could be 4d or 2d -
// Map {(c, k), p} -> {(p, h, w, k), (p, h, w, c)
// Map {(c, k), p} -> {(1, 1, p, k), (1, 1, p, c)
//
// mapping for TYPE_DEDX:
// operand B is internally transposed so -
// Map {(b, h, w, c), p} -> {(b, h, w, p), (0, 0, c, p)
//
// mapping for GEMM/FC nodes (TYPE_GEMM_DEDW/TYPE_GEMM_DEDX/TYPE_GEMM/TYPE_FC):
// {(w, k), p} ->  A transposed? (p, w) : (w, p) ,
//                 B transposed? (k, p) : (p, k)
SliceReferenceList MMEBackwardSliceMapping::getInputs(const SliceRefCommonDimIdxPair& slice) const
{
    pSliceReference outSliceRef = slice.first;
    unsigned commonDimSliceCoord = slice.second;

    pSliceReference refA(new SliceReference(m_inOperands[0]));
    fillCommonDim(OperandType::opA, refA, commonDimSliceCoord);
    fillNonCommonDim(OperandType::opA, refA, outSliceRef);

    pSliceReference refB(new SliceReference(m_inOperands[1]));
    fillCommonDim(OperandType::opB, refB, commonDimSliceCoord);
    fillNonCommonDim(OperandType::opB, refB, outSliceRef);

    // align the shape tensors slicing to the output tensor slicing
    auto mmeNode = std::dynamic_pointer_cast<MmeNode>(m_mmeNode);
    HB_ASSERT_PTR(mmeNode);
    pSlicedOperand biasOperand  = mmeNode->hasBias() ? m_inOperands.back() : nullptr;
    pSlicedOperand shapeOperand = m_inOperands.back()->originalTensor->isShapeTensor() ? m_inOperands.back() : nullptr;
    HB_ASSERT(!(biasOperand && shapeOperand),
              "same input cannot be shape tensor and bias {}",
              m_inOperands.back()->originalTensor->getName());
    if(shapeOperand)
    {
        pSliceReference refShapeTensor(new SliceReference(shapeOperand));
        refShapeTensor->coordinates = outSliceRef->coordinates; // copy sliced coordinates from output slice
        return {refA, refB, refShapeTensor};
    }
    else if (biasOperand)
    {
        HB_ASSERT(biasOperand->originalTensor->getDim() == 1,
                  "bias should be 1D but have {}",
                  biasOperand->originalTensor->getDim());
        pSliceReference refBiasTensor(new SliceReference(biasOperand));
        refBiasTensor->coordinates[DIM_C] =
            outSliceRef->coordinates[DIM_C];  // copy sliced coordinates from output slice
        return {refA, refB, refBiasTensor};
    }
    return {refA, refB};
}

void MMEBackwardSliceMapping::fillNonCommonDim(OperandType opType, const pSliceReference& inSliceRef,
                                               const pSliceReference& outSliceRef) const
{
    const DimVector* inputDims;
    const DimVector* outputDims;
    switch(opType)
    {
        case OperandType::opA:
            inputDims = &m_dimController.nonCommonDimOperandA();
            outputDims = &m_dimController.heightOutput();
            break;
        case OperandType::opB:
            inputDims = &m_dimController.nonCommonDimOperandB();
            outputDims = &m_dimController.widthOutput();
            break;
        default:
            SLC_ERR("Invalid operandType");
            return;
    }
    HB_ASSERT(inputDims->size() == outputDims->size(),
              "non common input part must have the same number of dimensions as the relevant part of output");
    for (auto inputIter = inputDims->begin(), outputIter = outputDims->begin();
         inputIter != inputDims->end() && outputIter != outputDims->end();
         ++inputIter, ++outputIter)
    {
        inSliceRef->coordinates[*inputIter] = outSliceRef->coordinates[*outputIter];
    }
}

/******************************************************************************
 * Example for 2 sliced common dims (h,b):
 * Given 2 slices on dim h and 3 slices on dim b,
 * coord will be between 0 and 5 (2*3-1).
 * Translation to a 2D point in [2,3] space.
 *            coordinates[h]   coordinates[b]
 * coord=0 :        0                0
 * coord=1 :        1                0
 * coord=2 :        0                1
 * coord=3 :        1                1
 * coord=4 :        0                2
 * coord=5 :        1                2
 * ----------------------------------------------------------------------------
 * Example for 3 sliced common dims (w,h,b):
 * Given 2 slices on dim w, 3 slices on dim h and 4 slices on dim b,
 * coord will be between 0 and 23 (2*3*4-1).
 * Translation to a 3D point in [2,3,4] space.
 *            coordinates[w]   coordinates[h]   coordinates[b]
 * coord=0 :        0                0                0
 * coord=1 :        1                0                0
 * coord=2 :        0                1                0
 * coord=3 :        1                1                0
 * coord=4 :        0                2                0
 * coord=5 :        1                2                0
 * coord=6 :        0                0                1
 * coord=7 :        1                0                1
 * coord=8 :        0                1                1
 * coord=9 :        1                1                1
 * coord=10:        0                2                1
 * coord=11:        1                2                1
 * coord=12:        0                0                2
 * coord=13:        1                0                2
 * coord=14:        0                1                2
 * coord=15:        1                1                2
 * coord=16:        0                2                2
 * coord=17:        1                2                2
 * coord=18:        0                0                3
 * coord=19:        1                0                3
 * coord=20:        0                1                3
 * coord=21:        1                1                3
 * coord=22:        0                2                3
 * coord=23:        1                2                3
 *****************************************************************************/
void MMEBackwardSliceMapping::fillCommonDim(OperandType opType, const pSliceReference& sliceRef, unsigned coord) const
{
    DimVector commonDims;
    switch(opType)
    {
        case OperandType::opA:
            commonDims = m_dimController.commonDimOperandA();
            break;
        case OperandType::opB:
            commonDims = m_dimController.commonDimOperandB();
            break;
        default:
        SLC_ERR("Invalid operandType");
            return;
    }

    // TODO: this should be refactored in SW-24633
    // slicedCommonDims ordered from fcd to external dims (w -> b) and don't have to be sequential
    std::vector<unsigned> slicedCommonDims;
    for(const auto dim : commonDims)
    {
        if(SlicedOperandUtils::isSlicedOnDimension(sliceRef->operand, dim))
        {
            slicedCommonDims.push_back(dim);
        }
    }
    auto numOfCommonDimsSliced = slicedCommonDims.size();
    HB_ASSERT(numOfCommonDimsSliced <= 3, "Currently more than 3 sliced common dims are not supported");

    if(numOfCommonDimsSliced == 0)
    {
        return; // Nothing to update
    }
    if(numOfCommonDimsSliced == 1)
    {
        sliceRef->coordinates[slicedCommonDims[0]] = coord;
    }
    else if (numOfCommonDimsSliced == 2)
    {
        // Walk on common dim from left to right (see example above)
        unsigned numOfSlicesPerFirstSlicedDim = SlicedOperandUtils::nofSlices(sliceRef->operand, slicedCommonDims[0]);
        unsigned firstDimCoord = coord % numOfSlicesPerFirstSlicedDim;
        unsigned secondDimCoord = coord / numOfSlicesPerFirstSlicedDim;
        sliceRef->coordinates[slicedCommonDims[0]] = firstDimCoord;
        sliceRef->coordinates[slicedCommonDims[1]] = secondDimCoord;
    }
    else  // numOfCommonDimsSliced == 3
    {
        // Walk on common dim from left to right (see example above)
        unsigned numOfSlicesPerFirstSlicedDim  = SlicedOperandUtils::nofSlices(sliceRef->operand, slicedCommonDims[0]);
        unsigned numOfSlicesPerSecondSlicedDim = SlicedOperandUtils::nofSlices(sliceRef->operand, slicedCommonDims[1]);
        // Stride for first dim is 1, repetition is every numOfSlicesPerFirstSlicedDim.
        unsigned firstDimCoord = coord % numOfSlicesPerFirstSlicedDim;
        // Stride for second dim is numOfSlicesPerFirstSlicedDim, repetition is every numOfSlicesPerSecondSlicedDim.
        unsigned secondDimCoord = (coord / numOfSlicesPerFirstSlicedDim) % numOfSlicesPerSecondSlicedDim;
        // Stride for third dim is mult of the 2 lower dims sizes, no repetition for external dim.
        unsigned thirdDimCoord = coord / (numOfSlicesPerFirstSlicedDim * numOfSlicesPerSecondSlicedDim);
        sliceRef->coordinates[slicedCommonDims[0]] = firstDimCoord;
        sliceRef->coordinates[slicedCommonDims[1]] = secondDimCoord;
        sliceRef->coordinates[slicedCommonDims[2]] = thirdDimCoord;
    }
}

SliceReferenceList MMETriviallyBatchedBackwardSliceMapping::getInputs(const SliceRefCommonDimIdxPair& slice) const
{
    auto res = MMEBackwardSliceMapping::getInputs(slice);

    const auto& coord = slice.first->coordinates;
    for (auto& in : res)
    {
        for (auto i = (unsigned) DIM_GEMM_BATCH; i < coord.size(); ++i)
        {
            if (in->operand->finalShape[i] > 1)
            {
                in->coordinates[i] = coord[i];
            }
            else
            {
                in->coordinates[i] = 0;
            }
        }
    }

    return res;
}

pBackwardSliceMapping MMESliceMapper::mapOutputToInputs(pNode          mmeNode,
                                                        pSlicedOperand inputA,
                                                        pSlicedOperand inputB,
                                                        pSlicedOperand output,
                                                        pSlicedOperand operandC)
{
    if (!mmeNode)
    {
        LOG_ERR(SRAM_SLICE, "Mapping output to input of null node");
        return nullptr;
    }
    std::vector<pSlicedOperand> inOperands({inputA, inputB});
    if (operandC)
    {
        inOperands.push_back(operandC);
    }
    return std::make_shared<MMEBackwardSliceMapping>(mmeNode, inOperands, output);
}

pBackwardSliceMapping MMETriviallyBatchedSliceMapper::mapOutputToInputs(pNode          mmeNode,
                                                                        pSlicedOperand inputA,
                                                                        pSlicedOperand inputB,
                                                                        pSlicedOperand output)
{
    if (!mmeNode)
    {
        LOG_ERR(SRAM_SLICE, "Mapping output to input of null node");
        return nullptr;
    }
    std::vector<pSlicedOperand> inOperands({inputA, inputB});
    return std::make_shared<MMETriviallyBatchedBackwardSliceMapping>(mmeNode, inOperands, output);
}

// the "trivial" slice mapping handles the basic functionality for 1 to 1  mapping (output to input)
// but also enables expansion of the functionality by inheriting from it and implementing getInputByOutput()
class TrivialBackwardSliceMapping : public BackwardSliceMapping
{
public:
    TrivialBackwardSliceMapping(const std::vector<pSlicedOperand>& inputs, const pSlicedOperand& output)
            : BackwardSliceMapping(inputs, output)
    {
    }

    virtual pBackwardSliceMapping clone(std::vector<pSlicedOperand>& inputOperands,
                                        pSlicedOperand& outputOperand) override
    {
        return std::make_shared<TrivialBackwardSliceMapping>(inputOperands, outputOperand);
    }

    SliceReferenceList getInputs(const SliceRefCommonDimIdxPair& slice) const override
    {
        SliceReferenceList  inputs;
        for (pSlicedOperand inputOperand : m_inOperands)
        {
            pSliceReference inputSlice = getInputByOutput(inputOperand, slice.first);
            inputs.push_back(inputSlice);
        }
        return inputs;
    }

    virtual pSliceReference getInputByOutput(const pSlicedOperand& inputOperand, const pSliceReference &slice) const
    {
        pSliceReference inputSlice = std::make_shared<SliceReference>(inputOperand);
        if (inputSlice->operand->originalTensor->getTensorType() == INPUT_DESCRIBING_SHAPE_TENSOR)
        {
            // We don't slice INPUT_DESCRIBING_SHAPE_TENSOR
            return inputSlice;
        }
        for (uint32_t coordIdx = 0; coordIdx < inputOperand->originalTensor->getDim(); ++coordIdx)
        {
            if (inputOperand->finalShape[coordIdx] == slice->operand->finalShape[coordIdx])
            {
                inputSlice->coordinates[coordIdx] = slice->coordinates[coordIdx];
            }
        }
        return inputSlice;
    }
};

// Map trivial reshape nodes from output to input, where each dimension in input/output is mapped to single or exact multiple
// of several dimensions in output/input
// Example: [C,W,H,B] can be mapped to [C, WH, B] or any other combination, as long that the sliced dim
// is not partial dimension (C/2 etc.)
class ReshapeBackwardSliceMapping : public BackwardSliceMapping
{
public:
    ReshapeBackwardSliceMapping(pSlicedOperand input, pSlicedOperand output)
            : BackwardSliceMapping({input}, output)
    {
        setPermutation(input->originalTensor, output->originalTensor);
    }

    virtual pBackwardSliceMapping clone(std::vector<pSlicedOperand>& inputOperands,
                                        pSlicedOperand& outputOperand) override
    {
        return std::make_shared<ReshapeBackwardSliceMapping>(inputOperands.front(), outputOperand);
    }

    SliceReferenceList getInputs(const SliceRefCommonDimIdxPair& slice) const override
    {
        SliceReferenceList  inputs;

        pSliceReference inputSlice = getInputByOutput(m_inOperands[0], slice.first);
        inputs.push_back(inputSlice);

        return inputs;
    }

    virtual pSliceReference getInputByOutput(const pSlicedOperand& inputOperand, const pSliceReference &slice) const
    {
        pSliceReference inputSlice = std::make_shared<SliceReference>(inputOperand);
        for (uint32_t coordIdx = 0; coordIdx < slice->operand->originalTensor->getDim(); ++coordIdx)
        {
            if (slice->coordinates[coordIdx] == 0) continue;
            Settable<uint32_t> inCoordinate;
            for (uint32_t idx = 0; idx < m_outputToInputDims[coordIdx].size(); ++idx)
            {
                uint32_t changedDim = m_outputToInputDims[coordIdx][idx];
                if (m_inOperands[0]->originalTensor->getSizeInElements(changedDim) > 1)
                {
                    HB_ASSERT(!inCoordinate.is_set(), "Two options to set coordinate");
                    inCoordinate.set(m_outputToInputDims[coordIdx][idx]);
                }
            }
            HB_ASSERT(inCoordinate.is_set(), "Can't find coordinate position");
            HB_ASSERT(inputSlice->coordinates[inCoordinate.value()] == 0, "Set same coordinate more then once - meaning not trivial reshape mapper");

            // Take the coordinate to the last dim reshaped
            inputSlice->coordinates[inCoordinate.value()] = slice->coordinates[coordIdx];
        }
        return inputSlice;
    }

protected:
    void setPermutation(const pTensor& input, const pTensor& output)
    {
        m_outputToInputDims = SlicedOperandUtils::getReshapeOutputToInputMapping(input->getShape(), output->getShape());
    }

    // Present what dim multiplication has done for the reshape
    SlicedOperandUtils::ReshapeOutputToInputMapping m_outputToInputDims;
};

class TPCBackwardSliceMapping : public TrivialBackwardSliceMapping
{
public:
    TPCBackwardSliceMapping(const std::vector<pSlicedOperand>& inputs, const pSlicedOperand& output)
    : TrivialBackwardSliceMapping(inputs, output)
    {
        for (const pSlicedOperand& input : m_inOperands)
        {
            m_inputIsBroadcast[input] =
                input->finalShape != m_outOperand->finalShape;
        }
    }

    virtual pSliceReference getInputByOutput(const pSlicedOperand& inputOperand,
                                             const pSliceReference &slice) const override
    {
        pSliceReference inputSlice = std::make_shared<SliceReference>(inputOperand);
        if (inputSlice->operand->originalTensor->getTensorType() == INPUT_DESCRIBING_SHAPE_TENSOR)
        {
            // We don't slice INPUT_DESCRIBING_SHAPE_TENSOR
            return inputSlice;
        }
        // TODO [SW-10295]: Separable but not elementwise effort should make this thing more generic
        for (unsigned dim = 0; dim < inputSlice->operand->originalTensor->getDim(); dim++)
        {
            if (inputSlice->operand->finalShape[dim] == slice->operand->finalShape[dim])
            {
                inputSlice->coordinates[dim] = slice->coordinates[dim];
            }
        }
        return inputSlice;
    }

protected:
    std::unordered_map<pSlicedOperand, bool> m_inputIsBroadcast;
};

class ForwardSliceMappingImpl : public ForwardSliceMapping
{
public:
    ForwardSliceMappingImpl(std::list<pSlicedOperand> allInputs,
                            std::list<pSlicedOperand> allOutputs)
    : m_slicedInputs(std::move(allInputs))
    , m_slicedOutputs(std::move(allOutputs))
    {}

    std::list<std::pair<SliceReferenceList, SliceReferenceList>> getInputsAndOutputs(const pSliceReference &input) override
    {
        const SliceReferenceList& inputReferences  = getAllSlicesByKeySlice(input, m_slicedInputs);
        const SliceReferenceList& outputReferences = getAllSlicesByKeySlice(input, m_slicedOutputs);

        return {{inputReferences, outputReferences}};
    }

    virtual pForwardSliceMapping clone(const std::list<pSlicedOperand>& inputs,
                                       const std::list<pSlicedOperand>& outputs) override
    {
        return std::make_shared<ForwardSliceMappingImpl>(inputs, outputs);
    }

    virtual const std::list<pSlicedOperand>& getInputs() const override { return m_slicedInputs; }
    virtual const std::list<pSlicedOperand>& getOutputs() const override { return m_slicedOutputs; }

protected:
    std::list<pSlicedOperand> m_slicedInputs;
    std::list<pSlicedOperand> m_slicedOutputs;

    SliceReferenceList getAllSlicesByKeySlice(const pSliceReference& keySlice,
                                              const std::list<pSlicedOperand>& slicedOperands) const
    {
        SliceReferenceList references;
        for (const pSlicedOperand& slicedOperand : slicedOperands)
        {
            if (slicedOperand == keySlice->operand)
            {
                references.push_back(keySlice);
            }
            else
            {
                pSliceReference sliceReference = std::make_shared<SliceReference>(slicedOperand);
                for (unsigned dim = 0; dim < slicedOperand->originalTensor->getDim(); dim++)
                {
                    if (SlicedOperandUtils::isSlicedOnDimension(slicedOperand, dim))
                    {
                        sliceReference->coordinates[dim] = keySlice->coordinates[dim];
                    }
                }
                references.push_back(sliceReference);
            }
        }
        return references;
    }
};

class NodeAccessPatternMapping
{
public:
    NodeAccessPatternMapping(const NodeAccessPatternPtr& accessPattern) : m_accessPattern(accessPattern) {}

    SliceReferenceList getMappedSlices(const pSliceReference& slice, const std::vector<pSlicedOperand>& operands) const
    {
        const auto& tensor = slice->operand->originalTensor;

        const TensorTile& sliceTile = getTileFromSlice(slice);
        const NodeTile&   nodeTile  = m_accessPattern->getNodeTile(tensor, sliceTile);

        SliceReferenceList mappedSlices;
        for (const pSlicedOperand& slicedOutput : operands)
        {
            mappedSlices.emplace_back(getSliceFromTile(nodeTile, slicedOutput));
        }

        return mappedSlices;
    }

    NodeAccessPatternPtr getAccessPattern() const { return m_accessPattern; }

private:
    TensorTile getTileFromSlice(const pSliceReference& slice) const
    {
        const auto& tensor = slice->operand->originalTensor;
        TensorTile  sliceTile(tensor->getDim(),
                             slice->operand->chunkDimensions,
                             SlicedOperandUtils::getSliceOffsets(slice));

        HB_ASSERT(validateSliceTile(sliceTile, tensor),
                  "Illegal slice tile for tensor '{}': geometry: [{}], offset: [{}]",
                  tensor->getName(),
                  toString(sliceTile.geometry, 'x'),
                  toString(sliceTile.offset, ','));

        return sliceTile;
    }

    // Validate that the slice inspired tile is itself a tiling of the tensor granularity
    bool validateSliceTile(const TensorTile& tensorTile, const TensorPtr& tensor) const
    {
        const TensorTile& granularity = m_accessPattern->getTensorGranularity(tensor);
        unsigned    tensorRank  = tensor->getDim();
        if (granularity.geometry.size() != tensorRank || granularity.offset.size() != tensorRank)
        {
            LOG_ERR(SRAM_SLICE,
                    "Granularity of tensor {} does not have the same dimensionality as the tensor. tensor rank: {}, "
                    "granularity sizes: geometry {}, offset {}",
                    tensor->getName(),
                    tensorRank,
                    granularity.geometry.size(),
                    granularity.offset.size());
            return false;
        }

        for (Dim dim = 0; dim < tensor->getDim(); dim++)
        {
            // optimization: avoid fancy validation where the tensor is not sliced
            if (tensorTile.geometry[dim] != tensor->getSizeInElements(dim) &&
                !validateSliceTileDim(tensorTile.geometry[dim],
                                      tensorTile.offset[dim],
                                      granularity.geometry[dim],
                                      granularity.offset[dim]))
            {
                LOG_ERR(SRAM_SLICE, "Invalid slice tile at dimension: {}", dim);
                return false;
            }
        }
        return true;
    }

    bool validateSliceTileDim(TensorTile::Size  tileSize,
                              TensorTile::Coord tileOffset,
                              TensorTile::Size  granularitySize,
                              TensorTile::Coord granularityOffset) const
    {
        // Expect the tile size to be a multiple of the granularity size:
        bool sizeValid = tileSize % granularitySize == 0;
        if (!sizeValid)
        {
            LOG_ERR(
                SRAM_SLICE,
                "Tensor tile size is {} and is not a multiple of the granularity in the dimension, which has size: {}",
                tileSize,
                granularitySize);
        }

        // Expect the tile offset to be a multiple of the granularity size as well, but offset by the granularity offset
        // i.e tileOffset = n * granularitySize + granularityOffset
        // For example, if the granularity offset is -5 and the granularity size is 10, then a slice starting at index 2
        // would have an offset of 2 * 10 - 5 = 15.
        // So when inversing the formula, we get that tileOffset - granularityOffset should be a multiple of the
        // granularitySize.
        bool offsetValid = (tileOffset - granularityOffset) % granularitySize == 0;
        if (!offsetValid)
        {
            LOG_ERR(SRAM_SLICE,
                    "Tensor offset is expected to be some integer n * {} + {}, but got {}",
                    granularitySize,
                    granularityOffset,
                    tileOffset);
        }

        return sizeValid && offsetValid;
    }

    pSliceReference getSliceFromTile(const NodeTile& nodeTile, const pSlicedOperand& slicedOperand) const
    {
        const TensorTile& tensorTile  = m_accessPattern->getTensorTile(slicedOperand->originalTensor, nodeTile);
        const TensorTile& granularity = m_accessPattern->getTensorGranularity(slicedOperand->originalTensor);

        auto sliceRef = std::make_shared<SliceReference>(slicedOperand);
        for (Dim dim = 0; dim < slicedOperand->originalTensor->getDim(); dim++)
        {
            // The sliceOffset = sliceCoord * sliceSize + operandOffset, namely:
            auto sliceDimOffset = tensorTile.offset[dim];
            auto sliceDimSize   = slicedOperand->chunkDimensions[dim];
            auto operandOffset  = granularity.offset[dim];

            TensorTile::Coord dimCoord;
            if ((sliceDimSize == 0) && (sliceDimOffset - operandOffset == 0) &&
                (sliceRef->operand->originalTensor->getTensorType() == INPUT_DESCRIBING_SHAPE_TENSOR))
            {
                // Allow zero slice size for input describing shape tensors when both numerator and denominator are zero
                dimCoord = 0;
            }
            else
            {
                // We want to divide by the chunk dimension
                HB_ASSERT(sliceDimSize != 0, "Slice size is 0 in dimension {}", dim);

                // To get the coord from the offset, need to inverse this equation:
                dimCoord = (sliceDimOffset - operandOffset) / sliceDimSize;
            }

            sliceRef->coordinates[dim] = dimCoord;
        }
        return sliceRef;
    }

    NodeAccessPatternPtr        m_accessPattern;
};

class NodeAccessPatternBackwardMapping : public BackwardSliceMapping
{
public:
    NodeAccessPatternBackwardMapping(const NodeAccessPatternPtr&        accessPattern,
                                     const std::vector<pSlicedOperand>& slicedInputs,
                                     const std::vector<pSlicedOperand>& slicedOutputs)
    : BackwardSliceMapping(slicedInputs, slicedOutputs.front()),
      m_accessPatternMapper(accessPattern),
      m_outputs(slicedOutputs)
    {
    }

    virtual const std::vector<pSlicedOperand>& getOutOperands() const { return m_outputs; }

    // Returns a list of SlicesReferences for all the inputs the produce the output slice
    SliceReferenceList getInputs(const SliceRefCommonDimIdxPair& outputSliceAndCDCounter) const override
    {
        HB_ASSERT(outputSliceAndCDCounter.second == 0, "Unexpected common dim counter > 0 for TPC output slice.");

        const auto& outputSlice = outputSliceAndCDCounter.first;
        return m_accessPatternMapper.getMappedSlices(outputSlice, getInOperands());
    }

    // Return a list of all output slices that match the given output slice.
    SliceReferenceList getOutputs(const pSliceReference& outputSlice) const override
    {
        return m_accessPatternMapper.getMappedSlices(outputSlice, getOutOperands());
    }

    pBackwardSliceMapping clone(std::vector<pSlicedOperand>& inputOperands, pSlicedOperand& output) override
    {
        HB_ASSERT(getOutOperands().size() == 1,
                  "Cloning node access pattern mapping is not supported for nodes with more than 1 output");
        return std::make_shared<NodeAccessPatternBackwardMapping>(m_accessPatternMapper.getAccessPattern(),
                                                                  inputOperands,
                                                                  std::vector<pSlicedOperand> {output});
    }

private:
    NodeAccessPatternMapping    m_accessPatternMapper;
    std::vector<pSlicedOperand> m_outputs;
};

class NodeAccessPatternForwardMapping : public ForwardSliceMapping
{
public:
    NodeAccessPatternForwardMapping(const NodeAccessPatternPtr&      accessPattern,
                                    const std::list<pSlicedOperand>& slicedInputs,
                                    const std::list<pSlicedOperand>& slicedOutputs)
    : ForwardSliceMapping(),
      m_accessPatternMapper(accessPattern),
      m_slicedInputs(slicedInputs),
      m_slicedOutputs(slicedOutputs),
      m_slicedInputsVec(slicedInputs.begin(), slicedInputs.end()),
      m_slicedOutputsVec(slicedOutputs.begin(), slicedOutputs.end())
    {
    }

    virtual const std::list<pSlicedOperand>& getInputs() const override { return m_slicedInputs; }

    virtual const std::list<pSlicedOperand>& getOutputs() const override { return m_slicedOutputs; }

    std::list<std::pair<SliceReferenceList, SliceReferenceList>>
    getInputsAndOutputs(const pSliceReference& input) override
    {
        const SliceReferenceList& inputReferences  = m_accessPatternMapper.getMappedSlices(input, m_slicedInputsVec);
        const SliceReferenceList& outputReferences = m_accessPatternMapper.getMappedSlices(input, m_slicedOutputsVec);

        return {{inputReferences, outputReferences}};
    }

    pForwardSliceMapping clone(const std::list<pSlicedOperand>& inputs,
                               const std::list<pSlicedOperand>& outputs) override
    {
        return std::make_shared<NodeAccessPatternForwardMapping>(m_accessPatternMapper.getAccessPattern(),
                                                                 inputs,
                                                                 outputs);
    }

private:
    NodeAccessPatternMapping    m_accessPatternMapper;
    std::list<pSlicedOperand>   m_slicedInputs;
    std::list<pSlicedOperand>   m_slicedOutputs;
    std::vector<pSlicedOperand> m_slicedInputsVec;
    std::vector<pSlicedOperand> m_slicedOutputsVec;
};

pBackwardSliceMapping TPCSliceMapper::mapOutputToInputs(pNode tpcNode,
                                                        std::list<pSlicedOperand> inputs,
                                                        pSlicedOperand output)
{
    std::vector<pSlicedOperand> inOperands(inputs.begin(), inputs.end());
    return std::make_shared<TPCBackwardSliceMapping>(inOperands, output);
}

pBackwardSliceMapping TrivialSliceMapper::mapOutputToInputs(pNode node,
                                                            std::list<pSlicedOperand> inputs,
                                                            pSlicedOperand output)
{
    std::vector<pSlicedOperand> inOperands(inputs.begin(), inputs.end());
    return std::make_shared<TrivialBackwardSliceMapping>(inOperands, output);
}

pForwardSliceMapping TrivialSliceMapper::mapSlicedOperandForward(const std::list<pSlicedOperand>& allInputs,
                                                                 const std::list<pSlicedOperand>& allOutputs)
{
    return std::make_shared<ForwardSliceMappingImpl>(allInputs, allOutputs);
}

pBackwardSliceMapping ReshapeSliceMapper::mapOutputToInput(pSlicedOperand input,
                                                           pSlicedOperand output)
{
    return std::make_shared<ReshapeBackwardSliceMapping>(input, output);
}

pBackwardSliceMapping AccessPatternSliceMapper::createBwdMapping(NodePtr                            node,
                                                                 const std::vector<pSlicedOperand>& allInputs,
                                                                 const std::vector<pSlicedOperand>& allOutputs)
{
    NodeAccessPatternPtr accessPattern = node->getNodeAccessPattern();
    HB_ASSERT_PTR(accessPattern);
    return std::make_shared<NodeAccessPatternBackwardMapping>(accessPattern, allInputs, allOutputs);
}

pForwardSliceMapping AccessPatternSliceMapper::createFwdMapping(NodePtr                          node,
                                                                const std::list<pSlicedOperand>& allInputs,
                                                                const std::list<pSlicedOperand>& allOutputs)
{
    NodeAccessPatternPtr accessPattern = node->getNodeAccessPattern();
    HB_ASSERT_PTR(accessPattern);
    return std::make_shared<NodeAccessPatternForwardMapping>(accessPattern, allInputs, allOutputs);
}