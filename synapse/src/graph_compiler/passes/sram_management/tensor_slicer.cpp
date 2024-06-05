#include "define_synapse_common.hpp"
#include "defs.h"
#include "log_manager.h"
#include "utils.h"
#include "slicing_utils.h"
#include "tensor_slicer.h"

static pTensor cloneTensor(const pTensor& tensor)
{
    return tensor->clone(false, false);
}

TensorSlicer::TensorSlicer(const Bundle::Solution::pSlicedOperand& slicedOperand, uint32_t bundleIdx, const Settable<uint64_t>& multiBufferId)
: m_sliceOperand(slicedOperand)
, m_baseName(slicedOperand->originalTensor->getName())
, m_originalTensorSize(slicedOperand->originalTensor->getAllSizesInElements())
, m_bundleIdx(bundleIdx)
, m_cache(slicedOperand, slicedOperand->originalTensor->getName())
, m_multiBufferId(multiBufferId)
{
}

bool TensorSlicer::alignTensorToCacheLine(const pTensor& tensor, const HalReader& halReader)
{
    if (GCFG_SRAM_SLICER_ALIGN_TO_CACHE_LINE.value())
    {
        const auto& originalStrides = tensor->getAllStridesInBytes();
        auto        alignedStrides  = SlicedOperandUtils::getCacheLineAlignedStrides(tensor->getAllSizesInElements(),
                                                                                     originalStrides,
                                                                                     tensor->getDim());
        if (alignedStrides)  // strides should be aligned
        {
            tensor->reshape(tensor->getDim(), tensor->getAllSizesInElements().data(), alignedStrides->data());

            // set device size manually - align it to cache line.
            // this is a work around for cases where "smaller" tensors cause higher fragmentation in allocation.
            uint64_t maxStride =
                *std::max_element(alignedStrides->begin(), alignedStrides->begin() + tensor->getDim() + 1);
            tensor->setDeviceSizeInBytes(maxStride);
            tensor->setIsRealInLogical(true);
            return true;
        }
    }
    return false;
}

pTensor TensorSlicer::getSlice(const HalReader& halReader, const CoordArray& coord, bool forceCreate)
{
    if (!forceCreate)
    {
        pTensor cachedTensor = m_cache.getCachedSlice(coord);
        if (cachedTensor != nullptr)
        {
            return cachedTensor;
        }
    }

    SizeArray sliceSize = SlicedOperandUtils::calcSliceSizesFromCoordinate(m_originalTensorSize, m_sliceOperand, coord);
    pTensor newSlice;
    newSlice = cloneTensor(m_sliceOperand->originalTensor);
    newSlice->reshape(newSlice->getDim(), sliceSize.data(), nullptr);
    newSlice->setName(getSliceName(coord));
    setSliceMemory(newSlice, m_sliceOperand->resideInSRAM || m_sliceOperand->originalTensor->inSram());
    // the following ensures the tensors have the intended data type by the brain. i.e. reduction tensors will be fp32
    if (m_sliceOperand->finalElementType != m_sliceOperand->originalTensor->getElementType())
    {
        newSlice->setElementType(m_sliceOperand->finalElementType);
    }
    bool tensorWasAligned = false;
    if (shouldAlignToCacheLine())
    {
        tensorWasAligned = alignTensorToCacheLine(newSlice, halReader);
    }
    m_cache.addNewSlice(coord, newSlice);

    LOG_DEBUG(BE_SLICER,
              "Create slice from tensor {}. coordinate [{}], size [{}], count {}, CLAlignOptimization = {}",
              m_baseName,
              toString(coord, ','),
              toString(sliceSize, ','),
              m_coordCount[coord],
              tensorWasAligned);
    m_cache.logCacheStatus();

    setMultiBufferId(newSlice);
    return newSlice;
}

SizeVector TensorSlicer::getSliceOffsets(const CoordArray& sliceCoord)
{
    SizeVector ret(m_sliceOperand->originalTensor->getDim(), 0);
    for (unsigned dim = 0; dim < m_sliceOperand->originalTensor->getDim(); dim++)
    {
        ret[dim] = SlicedOperandUtils::getSliceCoordOffset(sliceCoord[dim],
                                                           m_sliceOperand->chunkDimensions[dim],
                                                           m_sliceOperand->overlapElementsCount[dim],
                                                           m_sliceOperand->offsetBefore[dim]);
    }
    return ret;
}

void TensorSlicer::destroySlice(const CoordArray& coord)
{
    m_cache.destroySlice(coord);
}

std::string TensorSlicer::getSliceName(const CoordArray& coord)
{
    const uint32_t index = m_coordCount[coord]++;

    return fmt::format("{}_slice_{}__{}__bundle_{}", m_baseName, toString(coord, '_'), index, m_bundleIdx);
}

void TensorSlicer::setSliceMemory(pTensor& slice, bool inSram)
{
    if (inSram)
    {
        slice->getTensorAnnotation().memory.location = TENSOR_IN_SRAM;
        slice->setMemorySectionID(getSramMemoryID());
    }
    else
    {
        slice->getTensorAnnotation().nonPersistentSectionInfo.sectionId.unset();
        slice->getTensorAnnotation().nonPersistentSectionInfo.bufferingLevel.unset();
        slice->getTensorAnnotation().memory.location = TENSOR_IN_DRAM;
        slice->setMemorySectionID(MEMORY_ID_RESERVED_FOR_WORKSPACE);
    }
    slice->setMemoryDescriptor(synMemoryDescriptor(false));
}


bool TensorSlicer::shouldAlignToCacheLine()
{
    return m_sliceOperand->alignWithCacheLine;
}

void TensorSlicer::setMultiBufferId(pTensor slice)
{
    // See comment in BundleSlicer::getSliceTensor
    if ((m_sliceOperand->numOfBuffers > 1 || m_sliceOperand->sharedChainMultiBuf ||
         m_sliceOperand->isFirstSliceSmaller()) &&
        m_sliceOperand->resideInSRAM)
    {
        HB_ASSERT(m_multiBufferId.is_set(), "multi-buffer id isn't set for slice {}", slice->getName());
        slice->getTensorAnnotation().nonPersistentSectionInfo.sectionId.set(m_multiBufferId.value());
        unsigned numBuffers =
            m_sliceOperand->sharedChainMultiBuf ? c_sharedMultiBufLevel : m_sliceOperand->numOfBuffers;
        slice->getTensorAnnotation().nonPersistentSectionInfo.bufferingLevel.set(numBuffers);
        LOG_DEBUG(BE_SLICER,
                  "Set multibufferId {} numBuffers {} to slice {}",
                  m_multiBufferId.value(),
                  numBuffers,
                  slice->getName());
    }
}
