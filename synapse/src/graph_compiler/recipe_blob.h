#pragma once

#include "infra/defs.h"
#include <cstdint>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include "utils.h"
#include "settable.h"
#include "queue_command.h"

struct blob_t;
class RecipeAllocator;

// Holds various blob metadata that should not be serialized - rather it can be used elsewhere .e.g when creating the
// ECB commands
struct BlobMetaData
{
    unsigned sfgSobInc = 0;  // SOB value increment
    struct
    {
        unsigned target;       // SOB target value before reset
        unsigned targetXps;    // SOB target value before reset for Gaudi3 transpose pipe in the MME, unused elsewhere
        unsigned totalNumEngs; // total number of engines involved in the reset (across all engine types)
        bool     isCanceled;   // a flag indicating canceled ResetSob command due to joined gemm and xpose resets
    } resetSob = {0};
    struct Rollover
    {
        unsigned target;      // SOB target value before rollover
        unsigned targetXps;   // SOB target value before rollover for Gaudi3 transpose pipe in the MME, unused elsewhere
        bool     isCanceled;  // a flag indicating canceled rollover command due to joined gemm and xpose resets
    };

    std::vector<Rollover> rollovers;
    unsigned numDcoresSplit = 1;
};

class ShapeNode;

class RecipeBlob
{
public:
    RecipeBlob(const HabanaGraph* g);
    virtual ~RecipeBlob() = default;

    uint8_t*       reserveBytes(uint64_t numBytes); // invalidates previously returned write pointers
    void           addPaddingNOPs(unsigned numNOPs);
    const uint8_t* getBasePtr() const;
    uint64_t       sizeInBytes() const;
    bool           isPatchingBlob() const                { return m_isPatchingBlob; }
    bool           isWorkDistBlob() const                { return m_isWorkDistBlob; }
    bool           isContainingExe() const               { return m_containsExe;    }
    bool           isContainingSwtc() const              { return m_containsSwtc;   }
    bool           isContainingMonArm() const            { return m_containsMonArm; }
    bool           isContainingDynamicPatchPoint() const { return m_containsDynamicPatchPoint; }

    bool           canBeCompressed() const { return m_containsOnlyCompressibleDynamicPatchPoints; }
    void           setAsPatchingBlob()     { m_isPatchingBlob = true; }
    void           setAsWorkDistBlob()     { m_isWorkDistBlob = true; }
    void           setContainsExe()        { m_containsExe = true;    }
    void           setContainsSwtc()       { m_containsSwtc = true;   }
    void           setContainsMonArm()     { m_containsMonArm = true; }

    ShapeNode*     setContainsDynamicPatchPoint(QueueCommand* cmd);

    void           calcHash() const;

    uint64_t       getHash() const;
    void           serialize(blob_t* pBlob) const;
    void           print() const;
    uint64_t       getSerializedOffset() const { return m_serializedOffset; }  // offset in bytes in serialized buffer
    void           setSerializedOffset(uint64_t offset) { m_serializedOffset = offset; }
    unsigned       getBlobId() const { return m_blobId; }

    friend bool operator==(const RecipeBlob& lhs, const RecipeBlob& rhs)
    {
        if (!lhs.m_containsOnlyCompressibleDynamicPatchPoints || !rhs.m_containsOnlyCompressibleDynamicPatchPoints)
        {
            return false;
        }
        bool value = lhs.m_data == rhs.m_data && lhs.m_patchedTensorsInfo == rhs.m_patchedTensorsInfo;
        // We check the equality of the flag `isPatchingBlob` since we don't want
        // to optimize one of the blobs and leave the other, resulting in missing blobs
        HB_ASSERT(!value || lhs.m_isPatchingBlob == rhs.m_isPatchingBlob, "Equal blobs contents, but one should be patched and the other shouldn't");
        return value;
    }

    friend bool operator!=(const RecipeBlob& lhs, const RecipeBlob& rhs) { return !(lhs == rhs); }

    void registerQueCmdForDbg(QueueCommand* cmd) { m_queCmds4dbg.emplace_back(cmd); }
    void printQueCmds() const;

    Settable<BlobMetaData>& getBlobMetaData() { return m_metadata; }

    void addPatchedTensorInfo(const BasicFieldInfo::PatchedTensorInfo* patchedTensorInfo);
    void setContainsOnlyCompressibleDynamicPatchPoint(bool val);

private:
    RecipeBlob(const RecipeBlob&) = delete;
    RecipeBlob& operator=(const RecipeBlob&) = delete;

    const HabanaGraph*         m_graph;
    std::vector<uint8_t>       m_data;
    bool                       m_isPatchingBlob;
    bool                       m_isWorkDistBlob;
    bool                       m_containsExe;
    bool                       m_containsSwtc;
    bool                       m_containsMonArm;
    bool                       m_containsDynamicPatchPoint = false;
    mutable uint64_t           m_hash;
    mutable bool               m_isHashInvalid;  // indicate if we need to retigger hash calculation
    uint64_t                   m_serializedOffset;
    unsigned                   m_blobId;
    std::vector<QueueCommand*> m_queCmds4dbg;  // for debug: vector holding the QueueCommands composing this blob
    Settable<BlobMetaData>     m_metadata;

    // the following fields are used at generation time and are not serialized
    bool                                           m_containsOnlyCompressibleDynamicPatchPoints = true;
    std::vector<BasicFieldInfo::PatchedTensorInfo> m_patchedTensorsInfo;
};

struct BlobCommitInfo
{
    uint64_t index;
    bool     isPatching;
    bool     isReused;
    bool     isWDWithPatching;  // a work distribution blob is not a patching blob but may contain patchpoints
    unsigned blobId;
    Settable<BlobMetaData> md;
};

// This class handles the creation of executing blob and patching blob,
// and pushing them to a BlobContainer
class RecipeBlobContainer
{
public:
    RecipeBlobContainer(const HabanaGraph* g);
    virtual ~RecipeBlobContainer();

    RecipeBlob*               getPatchingBlob();
    RecipeBlob*               getExecutionBlob();
    RecipeBlob*               getWorkDistBlob();
    std::list<BlobCommitInfo> commitBlobs();

    void serialize(uint64_t*        pNumBlobs,
                   blob_t**         ppBlobs,
                   uint64_t*        pTotalBlobsSizeInBytes,
                   uint64_t**       executionBlobBuffer,
                   uint64_t*        executionBlobBufferSize,
                   uint64_t**       patchingBlobBuffer,
                   uint64_t*        patchingBlobBufferSize,
                   uint32_t**       workDistBlobBuffer,
                   uint64_t*        workDistBlobBufferSize,
                   RecipeAllocator* pRecipeAlloc) const;

    void                      addBlob(RecipeBlob* blob, uint64_t& blobIdx, bool& isReused); // public for testing purposes
    const RecipeBlob*         getBlobByIndex(uint64_t index) const { return m_commitedBlobs[index]; }
    uint64_t                  getBlobCount() const { return m_commitedBlobs.size(); }
    uint64_t                  print() const;

private:

    RecipeBlobContainer(const RecipeBlobContainer&) = delete;
    RecipeBlobContainer& operator=(const RecipeBlobContainer&) = delete;

    void calcBufSizeByBlobsChunks(RecipeBlob* blob,
                                  uint64_t*   pChunksNum,
                                  uint64_t*   sizeInBytes,
                                  size_t      blobsChunkSizeInBytes,
                                  uint64_t*   padding);

    const HabanaGraph*        m_graph;
    RecipeBlob*               m_currentPatchingBlob;
    RecipeBlob*               m_currentExecutionBlob;
    RecipeBlob*               m_currentWorkDistBlob;
    uint64_t                  m_patcBlobsSizeInBytes;
    uint64_t                  m_exeBlobsSizeInBytes;
    uint64_t                  m_workDistBlobsSizeInBytes;
    uint64_t                  m_patchingBlobsChunksNum;
    uint64_t                  m_executionBlobsChunksNum;
    uint64_t                  m_workDistBlobsChunksNum;
    uint64_t                  m_patchingBlobsChunkSize;
    uint64_t                  m_executionBlobsChunkSize;
    uint64_t                  m_totalExeBuffPadding;       // for statistics only
    uint64_t                  m_totalPatchBuffPadding;     // for statistics only
    uint64_t                  m_totalWorkDistBuffPadding;  // for statistics only
    std::vector<RecipeBlob*>  m_commitedBlobs;
    std::unordered_map<RecipeBlob*, uint64_t, SynapsePointerHasher, SynapsePointerEqualTo>  m_mapBlobToIndex;
};
