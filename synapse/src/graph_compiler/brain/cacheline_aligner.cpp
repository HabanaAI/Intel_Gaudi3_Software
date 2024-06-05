#include "cacheline_aligner.h"
#include "compilation_hal_reader.h"
#include "tensor.h"
#include "types.h"
#include "utils.h"
#include "brain_conf.h"

CachelineAligner::CachelineAligner(HabanaGraph& g, const BPTClonePersistenceMap& bptClonePersistence)
: m_graph(g),
  m_bptClonePersistence(bptClonePersistence),
  m_cachelineSize(CompilationHalReader::getHalReader()->getCacheLineSizeInBytes()),
  m_halfCachelineSize(m_cachelineSize / 2)
{
}

void CachelineAligner::run()
{
    for (const auto& n : m_graph.getNodes())
    {
        if (!n || !Node::isJoinNode(n)) continue;
        HB_ASSERT(n->getNumOutputs() == 1,
                  "Expecting join node {} to have 1 output (found: {})",
                  n->getNodeName(),
                  n->getNumOutputs());
        const auto& outputBPT = n->getOutput(0);

        // skip graph persistent bpts
        if (isGraphPersistent(outputBPT))
        {
            LOG_TRACE(LB_PARTIALS, "Skipping graph persistent output bpt {}", outputBPT->getName());
            continue;
        }

        // skip real logical bpts which's strides we cannot change
        if (outputBPT->isRealInLogical() || outputBPT->isRealInAliasing())
        {
            LOG_TRACE(LB_PARTIALS,
                      "Skipping real in logical: {} / real in aliasing: {} output BPT {}",
                      outputBPT->isRealInLogical(),
                      outputBPT->isRealInAliasing(),
                      outputBPT->getName());
            continue;
        }
        // check fcd stride alignment to cache line
        NStrideArray initialStrides;
        outputBPT->getNStridesInBytes(initialStrides.data());
        if (isAlignedFcdStride(initialStrides))
        {
            LOG_TRACE(LB_PARTIALS,
                      "Skipping aligned fcd stride {} output BPT {}",
                      initialStrides.at(1),
                      outputBPT->getName());
            continue;
        }

        const auto     sizes(outputBPT->getNSizesInElements());
        const unsigned alignTargetSize = getAlignmentTargetSize(initialStrides[FCD_DIM + 1]);
        const auto     alignedStrides =
            getCachelineAlignedStrides(outputBPT->getDim(), sizes, initialStrides, alignTargetSize);
        const auto unalignedSize = outputBPT->getTotalSizeInBytes();
        HB_ASSERT(unalignedSize > 0,
                  "Expecting bpt: {}, sizes: [{}], strides: [{}] to have a non zero size",
                  outputBPT->getName(),
                  toString(sizes, ','),
                  toString(initialStrides, ','));

        const auto alignedSize            = getAlignedTensorSizeInBytes(outputBPT, alignedStrides);
        const auto relativeMemoryIncrease = static_cast<float>(alignedSize - unalignedSize) / unalignedSize;
        if (relativeMemoryIncrease <= GCFG_MAX_RELATIVE_ALIGNMENT_INCREASE_RATIO.value())
        {
            alignFcdStrideToCacheLine(outputBPT, alignedStrides, alignedSize);
            LOG_DEBUG(LB_PARTIALS,
                      "Aligned bpt: {} to target size: {} B, FCD size: {} B, unaligned size: {} MB, aligned "
                      "size: {} MB, size increase relative to unaligned: {:>3.2f} %",
                      outputBPT->getName(),
                      alignTargetSize,
                      initialStrides[FCD_DIM + 1],
                      bToMb(unalignedSize),
                      bToMb(alignedSize),
                      relativeMemoryIncrease * 100);
            outputBPT->setName(fmt::format("{}_fcd_cl_aligned", outputBPT->getName()));
        }
        else
        {
            LOG_TRACE(LB_PARTIALS,
                      "Exceeded max memory increase, bpt: {}, FCD size: {} B, unaligned size: {} MB, aligned size: {} "
                      "MB, size increase relative to unaligned: {:>3.2f} %",
                      outputBPT->getName(),
                      initialStrides[FCD_DIM + 1],
                      bToMb(unalignedSize),
                      bToMb(alignedSize),
                      relativeMemoryIncrease * 100);
        }
    }
}

bool CachelineAligner::isAlignedFcdStride(const NStrideArray& strides) const
{
    const auto& fcdStride = strides.at(FCD_DIM + 1);
    return ((fcdStride % getCachelineSize()) == 0) || (fcdStride == getHalfCachelineSize());
}

bool CachelineAligner::isGraphPersistent(const TensorPtr& t) const
{
    // if t is a bpt clone, determine it's persistence via the persistence map
    // otherwise the indication on the tensor itself can be counted on
    const auto it = m_bptClonePersistence.find(t);
    return it == m_bptClonePersistence.end() ? t->isUserManagedDram() : it->second;
}

void CachelineAligner::alignFcdStrideToCacheLine(const TensorPtr&    bpt,
                                                 const NStrideArray& alignedStrides,
                                                 TSize               alignedSize)
{
    bpt->reshape(bpt->getDim(),
                 nullptr /*max size doesn't change*/,
                 alignedStrides.data(),
                 nullptr /*min size doesn't change*/);
    bpt->setDeviceSizeInBytes(alignedSize);
    bpt->setIsRealInLogical(true);  // enforce aligned strides will be set to the bpt slices in handle logical ops
}

NStrideArray::value_type CachelineAligner::getAlignedTensorSizeInBytes(const TensorPtr&    bpt,
                                                                       const NStrideArray& alignedStrides) const
{
    return *std::max_element(alignedStrides.begin(), alignedStrides.begin() + bpt->getDim() + 1);
}

NStrideArray CachelineAligner::getCachelineAlignedStrides(unsigned            rank,
                                                          const NSizeArray&   sizes,
                                                          const NStrideArray& initialStrides,
                                                          unsigned            alignTargetSize) const
{
    NStrideArray alignedStrides(initialStrides);
    const auto   alignedFcdStride = round_to_multiple(initialStrides[FCD_DIM + 1], alignTargetSize);
    alignedStrides[FCD_DIM + 1]   = alignedFcdStride;
    for (auto dim = (FCD_DIM + 1); dim < rank; ++dim)
    {
        alignedStrides[dim + 1] = alignedStrides[dim] * sizes[dim];
    }
    return alignedStrides;
}

unsigned CachelineAligner::getAlignmentTargetSize(const NStrideArray::value_type& fcdStrideInBytes) const
{
    // align to full cachline
    UNUSED(fcdStrideInBytes);
    return getCachelineSize();
}

unsigned SmallFCDHalfCachelineAligner::getAlignmentTargetSize(const NStrideArray::value_type& fcdStrideInBytes) const
{
    // if FCD stride < CL/2: align to half cacheline to reduce memory increase
    const auto target = fcdStrideInBytes < getHalfCachelineSize() ? getHalfCachelineSize() : getCachelineSize();
    return target;
}