#include "recipe_blob.h"

#include "habana_global_conf.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "infra/defs.h"
#include "infra/fasthash.h"
#include "recipe.h"
#include "recipe_allocator.h"

#include <limits>

//-----------------------------------------------------------------------------
//                            RecipeBlob
//-----------------------------------------------------------------------------

RecipeBlob::RecipeBlob(const HabanaGraph* g)
: m_graph(g),
  m_isPatchingBlob(false),
  m_isWorkDistBlob(false),
  m_containsExe(false),
  m_containsSwtc(false),
  m_containsMonArm(false),
  m_hash(0),
  m_isHashInvalid(true),
  m_serializedOffset(0)
{
    thread_local static unsigned blobUniqueId = 1;

    // m_blobId is a blob unique identifier used to associate blob with its dynamic patch points
    m_blobId = blobUniqueId++;
}

uint8_t* RecipeBlob::reserveBytes(uint64_t numBytes)
{
    uint64_t initialSize = m_data.size();
    m_data.insert(m_data.end(), numBytes, 0);
    m_isHashInvalid = true; // retrigger hash calculation
    return m_data.data() + initialSize;
}

void RecipeBlob::addPaddingNOPs(unsigned numNOPs)
{
    HB_ASSERT(!isWorkDistBlob(), "we shouldn't pad with NOPs the work-distribution blob");
    if (numNOPs == 0) return;
    const QueueCommandPtr nop  = m_graph->getCodeGenerator()->getCommandFactory().getNop();
    uint8_t*        pBuf = reserveBytes(numNOPs * nop->GetBinarySize());
    for (unsigned i = 0; i < numNOPs; ++i)
    {
        nop->writeInstruction(pBuf);
        pBuf += nop->GetBinarySize();
    }
    m_isHashInvalid = true; // retrigger hash calculation
}

const uint8_t* RecipeBlob::getBasePtr() const
{
    return m_data.data();
}

uint64_t RecipeBlob::sizeInBytes() const
{
    return m_data.size();
}

void RecipeBlob::calcHash() const
{
    m_hash          = fasthash(m_data.data(), m_data.size());
    m_isHashInvalid = false;
}

uint64_t RecipeBlob::getHash() const
{
    if (m_isHashInvalid)
    {
        calcHash();
    }
    return m_hash;
}

ShapeNode* RecipeBlob::setContainsDynamicPatchPoint(QueueCommand* cmd)
{
    ShapeNode* ret = nullptr;
    m_containsDynamicPatchPoint = true;
    bool onlyCompressible = cmd->getBasicFieldsContainerInfo().hasOnlyCompressibleDynamicPatchPoints();
    setContainsOnlyCompressibleDynamicPatchPoint(onlyCompressible);
    if (canBeCompressed())
    {
        for (const auto& fld : cmd->getBasicFieldsContainerInfo().retrieveBasicFieldInfoSet())
        {
            addPatchedTensorInfo(fld.second->getPatchedTensorInfo());
            if (ret == nullptr)
            {
                ret = fld.second->getOrigin()->getShapeNode();
            }
        }
    }
    return ret;
}


void RecipeBlob::serialize(blob_t* pBlob) const
{
    if (isPatchingBlob())
    {
        pBlob->blob_type_all = blob_t::PATCHING;
    }
    else if (isWorkDistBlob())
    {
        pBlob->blob_type_all = blob_t::DYNAMIC;
    }
    else
    {
        pBlob->blob_type_all = blob_t::EXE;
    }

    pBlob->size = m_data.size();
    memcpy(pBlob->data, m_data.data(), pBlob->size);
}

void RecipeBlob::print() const
{
    LOG_DEBUG(RECIPE_GEN,
              "      blob type = {}",
              isWorkDistBlob() ? "dynamic-WD" : isPatchingBlob() ? "static-patching" : "static-executable");
    LOG_DEBUG(RECIPE_GEN, "      blob size in bytes = {}", m_data.size());
}

void RecipeBlob::printQueCmds() const
{
    std::for_each(m_queCmds4dbg.begin(), m_queCmds4dbg.end(), [](QueueCommand* c) { c->Print(); });
}

void RecipeBlob::addPatchedTensorInfo(const BasicFieldInfo::PatchedTensorInfo* patchedTensorInfo)
{
    if (patchedTensorInfo != nullptr)
    {
        m_patchedTensorsInfo.push_back(*patchedTensorInfo);
    }
}

void RecipeBlob::setContainsOnlyCompressibleDynamicPatchPoint(bool val)
{
    m_containsOnlyCompressibleDynamicPatchPoints = m_containsOnlyCompressibleDynamicPatchPoints && val;
    if (!m_containsOnlyCompressibleDynamicPatchPoints)
    {
        m_patchedTensorsInfo.clear();
    }
}

//-----------------------------------------------------------------------------
//                            RecipeBlobContainer
//-----------------------------------------------------------------------------
uint64_t PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES = 128 * 1024;

RecipeBlobContainer::RecipeBlobContainer(const HabanaGraph* g)
: m_graph(g),
  m_currentPatchingBlob(nullptr),
  m_currentExecutionBlob(nullptr),
  m_currentWorkDistBlob(nullptr),
  m_patcBlobsSizeInBytes(0),
  m_exeBlobsSizeInBytes(0),
  m_workDistBlobsSizeInBytes(0),
  m_patchingBlobsChunksNum(1),
  m_executionBlobsChunksNum(1),
  m_workDistBlobsChunksNum(1),
  m_patchingBlobsChunkSize(PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES),
  m_executionBlobsChunkSize(EXECUTION_BLOBS_CHUNK_SIZE_IN_BYTES),
  m_totalExeBuffPadding(0),
  m_totalPatchBuffPadding(0),
  m_totalWorkDistBuffPadding(0)
{
}

RecipeBlobContainer::~RecipeBlobContainer()
{
    for (RecipeBlob* pBlob : m_commitedBlobs)
    {
        delete pBlob;
    }
    delete m_currentPatchingBlob;
    delete m_currentExecutionBlob;
    delete m_currentWorkDistBlob;
}

RecipeBlob* RecipeBlobContainer::getPatchingBlob()
{
    if (m_currentPatchingBlob == nullptr)
    {
        m_currentPatchingBlob = new RecipeBlob(m_graph);
        m_currentPatchingBlob->setAsPatchingBlob();
    }
    return m_currentPatchingBlob;
}

RecipeBlob* RecipeBlobContainer::getExecutionBlob()
{
    if (m_currentExecutionBlob == nullptr)
    {
        m_currentExecutionBlob = new RecipeBlob(m_graph);
    }
    return m_currentExecutionBlob;
}

RecipeBlob* RecipeBlobContainer::getWorkDistBlob()
{
    if (m_currentWorkDistBlob == nullptr)
    {
        m_currentWorkDistBlob = new RecipeBlob(m_graph);
        m_currentWorkDistBlob->setAsWorkDistBlob();
    }
    return m_currentWorkDistBlob;
}

std::list<BlobCommitInfo> RecipeBlobContainer::commitBlobs()
{
    std::list<BlobCommitInfo> ret;
    std::vector<RecipeBlob*>  stagedBlobs;
    BlobCommitInfo            commitInfo = {0};
    auto                      exeItr     = ret.end();
    auto                      swtcItr    = ret.end();
    auto                      monArmItr  = ret.end();

    // Put the used blobs in the staging list and reset their pointers
    if (m_currentPatchingBlob)  stagedBlobs.emplace_back(m_currentPatchingBlob);
    if (m_currentExecutionBlob) stagedBlobs.emplace_back(m_currentExecutionBlob);
    if (m_currentWorkDistBlob)  stagedBlobs.emplace_back(m_currentWorkDistBlob);
    m_currentPatchingBlob  = nullptr;
    m_currentExecutionBlob = nullptr;
    m_currentWorkDistBlob  = nullptr;

    // Add each blob to the container and build the returned blob info list
    for (RecipeBlob* b : stagedBlobs)
    {
        bool exeBlob    = b->isContainingExe();
        bool swtcBlob   = b->isContainingSwtc() && !exeBlob;  // the exe blob may also have switch
        bool monArmBlob = b->isContainingMonArm() && !exeBlob;

        commitInfo.isPatching       = b->isPatchingBlob();
        commitInfo.isWDWithPatching = b->isWorkDistBlob() && b->isContainingDynamicPatchPoint();
        commitInfo.blobId           = b->getBlobId();  // associate blob unique identifier with the blob index

        // propagate blob metatdata
        commitInfo.md = b->getBlobMetaData();
        b->getBlobMetaData().unset();

        HB_ASSERT(!monArmBlob || (monArmBlob && !swtcBlob), "blob with monitor arm cannot contain switch CQ");

        addBlob(b, commitInfo.index, commitInfo.isReused);

        // Order matters! The order of the blobs returned by this function is the order that they will be executed.
        // Process staged blobs according to their staging order while applying the following ordering rules:
        //   1. blob containing the exe command - goes last
        //   2. blob containing static switch CQ (not exe switching) - goes just before the exe blob
        //   3. blob containing the monitor arm command w/o the exe (if there is one) - goes first
        //   4. blob containing patchable commands (patching blob) - goes first (but second to the previous bullet)
        // Assumptions:
        //   1. blob containing the monitor arm command cannot contain switch CQ as they belong to different platforms
        if (exeBlob)
        {
            exeItr = ret.insert(ret.end(), commitInfo);  // exe must go last
        }
        else if (swtcBlob)
        {
            swtcItr = ret.insert(exeItr, commitInfo);  // insert before the exe
        }
        else if (monArmBlob)
        {
            ret.push_front(commitInfo);  // mon arm goes first
            monArmItr = ret.begin();
        }
        else if (commitInfo.isPatching)
        {
            if (monArmItr == ret.end())
            {
                ret.push_front(commitInfo);
            }
            else
            {
                monArmItr++;
                ret.insert(monArmItr--, commitInfo);
            }
        }
        else if (swtcItr != ret.end())
        {
            ret.insert(swtcItr, commitInfo);  // insert before the swtc
        }
        else
        {
            ret.insert(exeItr, commitInfo);  // insert before the exe (i.e. before the last)
        }
    }
    return ret;
}

void RecipeBlobContainer::calcBufSizeByBlobsChunks(RecipeBlob* blob,
                                                   uint64_t*   pChunksNum,
                                                   uint64_t*   sizeInBytes,
                                                   size_t      blobsChunkSizeInBytes,
                                                   uint64_t*   padding)
{
    HB_ASSERT(blob != nullptr && pChunksNum != nullptr && sizeInBytes != nullptr, "got input null pointers");

    // In this case the Blobs Chunk Mechanism is inactive
    if (blobsChunkSizeInBytes == 0)
    {
        *sizeInBytes += blob->sizeInBytes();
        return;
    }

    // Verify that blob size is not larger than blobs chunk size
    HB_ASSERT(blob->sizeInBytes() <= blobsChunkSizeInBytes, "Current blob's size is bigger than blobs chunk size");

    if (*sizeInBytes + blob->sizeInBytes() > *pChunksNum * blobsChunkSizeInBytes)
    {
        *padding += *pChunksNum * blobsChunkSizeInBytes - *sizeInBytes;
        *sizeInBytes = *pChunksNum * blobsChunkSizeInBytes;
        *pChunksNum = *pChunksNum + 1;
    }
    else if (*sizeInBytes + blob->sizeInBytes() == *pChunksNum * blobsChunkSizeInBytes)
    {
        *pChunksNum = *pChunksNum + 1;
    }
    *sizeInBytes += blob->sizeInBytes();
}

void RecipeBlobContainer::addBlob(RecipeBlob* blob, uint64_t& blobIdx, bool& isReused)
{
    HB_ASSERT_PTR(blob);

    // Pad with NOPs if needed, but only for the static blobs (work-dist blob doesn't contain QMAN commands)
    if (!blob->isWorkDistBlob())
    {
        unsigned blobSizeGranularityInBytes = m_graph->getHALReader()->getCpDmaAlignment();
        unsigned blobSizeReminder           = calcPaddingSize(blob->sizeInBytes(), blobSizeGranularityInBytes);

        // Pad with NOPs until blob size is a multiple of blobSizeGranularityInBytes
        if (blobSizeReminder != 0)
        {
            blob->addPaddingNOPs(blobSizeReminder / 8);  // div by nop size (8 bytes)
        }
    }

    if (!GCFG_COMPRESS_BLOBS.value() || !blob->canBeCompressed())
    {
        m_commitedBlobs.emplace_back(blob);
        blobIdx = m_commitedBlobs.size() - 1;
        isReused = false;
    }
    else
    {
        // else, compress
        blob->calcHash();

        // Register the blob unless it is identical to a previous one and thus can be deleted
        // The map handles calculating the hash and equality testing.
        auto blobIt = m_mapBlobToIndex.find(blob);
        if (blobIt == m_mapBlobToIndex.end())
        {
            m_commitedBlobs.emplace_back(blob);
            blobIdx = m_commitedBlobs.size() - 1;
            m_mapBlobToIndex[blob] = blobIdx;
            isReused = false;
        }
        else
        {
            blobIdx = blobIt->second;
            isReused = true;
            if (LOG_LEVEL_AT_LEAST_TRACE(RECIPE_GEN))
            {
                LOG_TRACE(RECIPE_GEN,
                          "{}: Optimizing blobs, dropping {} blob since identical hash was found in blob at index {}",
                          HLLOG_FUNC,
                          blob->isWorkDistBlob()   ? "dynamic-WD"
                          : blob->isPatchingBlob() ? "patching"
                                                   : "execution",
                          blobIdx);
            }

            delete blob;
        }
    }

    // Accumulate blob size and record its serialized offset only if not reused
    if (!isReused)
    {
        if (blob->isPatchingBlob())
        {
            calcBufSizeByBlobsChunks(blob,
                                     &m_patchingBlobsChunksNum,
                                     &m_patcBlobsSizeInBytes,
                                     m_patchingBlobsChunkSize,
                                     &m_totalPatchBuffPadding);

            // record the offset of the blob within the serialized buffer after chunk padding
            blob->setSerializedOffset(m_patcBlobsSizeInBytes - blob->sizeInBytes());
        }
        else if (blob->isWorkDistBlob())
        {
            calcBufSizeByBlobsChunks(blob,
                                     &m_workDistBlobsChunksNum,
                                     &m_workDistBlobsSizeInBytes,
                                     m_patchingBlobsChunkSize,
                                     &m_totalWorkDistBuffPadding);

            // record the offset of the blob within the serialized buffer after chunk padding
            blob->setSerializedOffset(m_workDistBlobsSizeInBytes - blob->sizeInBytes());
        }
        else
        {
            calcBufSizeByBlobsChunks(blob,
                                     &m_executionBlobsChunksNum,
                                     &m_exeBlobsSizeInBytes,
                                     m_executionBlobsChunkSize,
                                     &m_totalExeBuffPadding);

            // record the offset of the blob within the serialized buffer after chunk padding
            blob->setSerializedOffset(m_exeBlobsSizeInBytes - blob->sizeInBytes());
        }
    }
}

void RecipeBlobContainer::serialize(uint64_t*        pNumBlobs,
                                    blob_t**         ppBlobs,
                                    uint64_t*        pTotalBlobsSizeInBytes,
                                    uint64_t**       executionBlobBuffer,
                                    uint64_t*        executionBlobBufferSize,
                                    uint64_t**       patchingBlobBuffer,
                                    uint64_t*        patchingBlobBufferSize,
                                    uint32_t**       workDistBlobBuffer,
                                    uint64_t*        workDistBlobBufferSize,
                                    RecipeAllocator* pRecipeAlloc) const
{
    HB_ASSERT_PTR(pNumBlobs);
    HB_ASSERT_PTR(ppBlobs);
    HB_ASSERT_PTR(pTotalBlobsSizeInBytes);
    HB_ASSERT_PTR(executionBlobBuffer);
    HB_ASSERT_PTR(executionBlobBufferSize);
    HB_ASSERT_PTR(patchingBlobBuffer);
    HB_ASSERT_PTR(patchingBlobBufferSize);
    HB_ASSERT_PTR(workDistBlobBuffer);
    HB_ASSERT_PTR(workDistBlobBufferSize);
    HB_ASSERT_PTR(pRecipeAlloc);

    *pNumBlobs              = m_commitedBlobs.size();
    *ppBlobs                = (blob_t*)pRecipeAlloc->allocate(*pNumBlobs * sizeof(blob_t));
    blob_t* pFiller         = *ppBlobs;
    *pTotalBlobsSizeInBytes = 0;

    // Allocate the execution buffer
    unsigned buffSizeU64 = CEIL(m_exeBlobsSizeInBytes, sizeof(uint64_t));
    *executionBlobBuffer = (uint64_t*)pRecipeAlloc->allocate(sizeof(uint64_t) * buffSizeU64, true);
    if (*executionBlobBuffer) memset(*executionBlobBuffer, 0, buffSizeU64 * sizeof(uint64_t));

    // Allocate the patching buffer
    buffSizeU64         = CEIL(m_patcBlobsSizeInBytes, sizeof(uint64_t));
    *patchingBlobBuffer = (uint64_t*)pRecipeAlloc->allocate(sizeof(uint64_t) * buffSizeU64, true);
    if (*patchingBlobBuffer) memset(*patchingBlobBuffer, 0, buffSizeU64 * sizeof(uint64_t));

    // Allocate the work distribution buffer
    unsigned buffSizeU32 = CEIL(m_workDistBlobsSizeInBytes, sizeof(uint32_t));
    *workDistBlobBuffer  = (uint32_t*)pRecipeAlloc->allocate(sizeof(uint32_t) * buffSizeU32, true);
    if (*workDistBlobBuffer) memset(*workDistBlobBuffer, 0, buffSizeU32 * sizeof(uint32_t));

    *executionBlobBufferSize = m_exeBlobsSizeInBytes;
    *patchingBlobBufferSize  = m_patcBlobsSizeInBytes;
    *workDistBlobBufferSize  = m_workDistBlobsSizeInBytes;

    for (auto blob : m_commitedBlobs)
    {
        char* pBuff = (char*)(*executionBlobBuffer);
        if (blob->isPatchingBlob()) pBuff = (char*)(*patchingBlobBuffer);
        if (blob->isWorkDistBlob()) pBuff = (char*)(*workDistBlobBuffer);
        pFiller->data = pBuff + blob->getSerializedOffset();
        blob->serialize(pFiller);
        *pTotalBlobsSizeInBytes += pFiller->size;  // accumulate blob sizes not including padding
        pFiller++;
    }

    // Log statistics
    LOG_DEBUG(RECIPE_GEN,
              "    Padding precentage: execution buffer = {}%, patching buffer = {}%, workDist buffer = {}%",
              m_exeBlobsSizeInBytes ? ((float)m_totalExeBuffPadding / m_exeBlobsSizeInBytes) * 100 : 0,
              m_patcBlobsSizeInBytes ? ((float)m_totalPatchBuffPadding / m_patcBlobsSizeInBytes) * 100 : 0,
              m_workDistBlobsSizeInBytes ? ((float)m_totalWorkDistBuffPadding / m_workDistBlobsSizeInBytes) * 100 : 0);
}

uint64_t RecipeBlobContainer::print() const
{
    LOG_DEBUG(RECIPE_GEN, "  Blob Container Dump:");
    LOG_DEBUG(RECIPE_GEN, "    Number of blobs = {}", m_commitedBlobs.size());
    uint64_t i         = 0;
    uint64_t totalSize = 0;
    for (auto blob : m_commitedBlobs)
    {
        LOG_DEBUG(RECIPE_GEN, "    Blob at index {}:", i++);
        blob->print();
        totalSize += blob->sizeInBytes();
    }

    if (LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN))
    {
        uint64_t sizeOfPatching =
            std::accumulate(m_commitedBlobs.begin(), m_commitedBlobs.end(), 0, [](uint64_t size, RecipeBlob* blob) {
                return size + (blob->isPatchingBlob() ? blob->sizeInBytes() : 0);
            });
        uint64_t numberOfPatching =
            std::count_if(m_commitedBlobs.begin(), m_commitedBlobs.end(), [](RecipeBlob* blob) {
                return blob->isPatchingBlob();
            });
        uint64_t sizeOfWorkDist =
            std::accumulate(m_commitedBlobs.begin(), m_commitedBlobs.end(), 0, [](uint64_t size, RecipeBlob* blob) {
                return size + (blob->isWorkDistBlob() ? blob->sizeInBytes() : 0);
            });
        uint64_t numberOfWorkDist = std::count_if(m_commitedBlobs.begin(), m_commitedBlobs.end(), [](RecipeBlob* blob) {
            return blob->isWorkDistBlob();
        });

        uint64_t sizeOfExecution   = totalSize - sizeOfPatching - sizeOfWorkDist;
        uint64_t numberOfExecution = m_commitedBlobs.size() - numberOfPatching - numberOfWorkDist;
        LOG_DEBUG(RECIPE_GEN, "    Execution blobs: Count: {}, Size: {}", numberOfExecution, sizeOfExecution);
        LOG_DEBUG(RECIPE_GEN, "    Patching blobs: Count: {}, Size: {}", numberOfPatching, sizeOfPatching);
        LOG_DEBUG(RECIPE_GEN, "    WorkDist blobs: Count: {}, Size: {}", numberOfWorkDist, sizeOfWorkDist);
    }
    return totalSize;
}
