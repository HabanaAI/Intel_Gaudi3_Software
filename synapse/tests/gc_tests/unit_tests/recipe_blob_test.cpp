#include "recipe_test_base.h"
#include "recipe.h"
#include "graph_compiler/recipe_blob.h"
#include "recipe_allocator.h"
#include "habana_graph.h"

static auto isPatchingLambda  = [](BlobCommitInfo bi){ return  bi.isPatching; };
static auto isExeLambda       = [](BlobCommitInfo bi){ return !bi.isPatching; };

void RecipeTestBase::blob_basic(HabanaGraph *g)
{
    const uint64_t blobSizeBytes = 32;
    RecipeBlob     blob(g);
    blob_t         serializedBlob = {0};

    ASSERT_EQ(blob.sizeInBytes(), 0);
    ASSERT_FALSE(blob.isPatchingBlob());
    blob.setAsPatchingBlob();
    ASSERT_TRUE(blob.isPatchingBlob());

    uint8_t* writePtr = blob.reserveBytes(blobSizeBytes);
    ASSERT_TRUE(writePtr != nullptr);
    ASSERT_EQ(blob.sizeInBytes(), blobSizeBytes);
    ASSERT_EQ(writePtr - blob.getBasePtr(), 0);
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        *writePtr++ = i;
    }

    serializedBlob.data = new char[blobSizeBytes];

    // test the serialize func
    blob.serialize(&serializedBlob);
    ASSERT_TRUE(serializedBlob.blob_type.requires_patching);
    ASSERT_EQ(serializedBlob.size, blobSizeBytes);
    uint8_t* readPtr = (uint8_t*)serializedBlob.data;
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        ASSERT_EQ(*readPtr++, i);
    }
    delete[] (uint64_t*)serializedBlob.data;

    // ensure blob remains consecutive as it grows big
    for (unsigned i = 1; i < 500000; i++) // we already reserved one batch of 32 bytes
    {
        writePtr = blob.reserveBytes(blobSizeBytes);
        ASSERT_EQ(writePtr, blob.getBasePtr() + (i * blobSizeBytes));
    }
    ASSERT_EQ(blob.sizeInBytes(), blobSizeBytes * 500000);
}

void RecipeTestBase::blob_container_basic(HabanaGraph *g)
{
    const uint64_t       blobSizeBytes = 24; // intentionally not a multiple of 32, but must be a multiple of 8
    RecipeBlobContainer  container(g);
    RecipeBlob*          blob1 = container.getPatchingBlob();
    RecipeBlob*          blob2 = container.getExecutionBlob();

    ASSERT_TRUE(blob1 != nullptr);
    ASSERT_TRUE(blob2 != nullptr);
    ASSERT_TRUE(blob1->isPatchingBlob());
    uint8_t* writePtr = blob1->reserveBytes(blobSizeBytes);
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        *writePtr++ = i;
    }
    writePtr = blob2->reserveBytes(blobSizeBytes);
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        *writePtr++ = i * 2;
    }

    std::list<BlobCommitInfo> commitedBlobs = container.commitBlobs();

    uint64_t  numBlobs;
    blob_t*   pBlobs;
    uint64_t  totalBlobsSizeInBytes;
    blob_t*   serializedBlob1;
    blob_t*   serializedBlob2;

    uint64_t* executionBlobBuffer;
    uint64_t* patchingBlobBuffer;
    uint32_t* workDistBlobBuffer;
    uint64_t  executionBlobBufferSize;
    uint64_t  patchingBlobBufferSize;
    uint64_t  workDistBlobBufferSize;

    RecipeAllocator recipeAlloc;

    container.serialize(&numBlobs,
                        &pBlobs,
                        &totalBlobsSizeInBytes,
                        &executionBlobBuffer,
                        &executionBlobBufferSize,
                        &patchingBlobBuffer,
                        &patchingBlobBufferSize,
                        &workDistBlobBuffer,
                        &workDistBlobBufferSize,
                        &recipeAlloc);

    uint blobAlginment = g->getHALReader()->getCpDmaAlignment(); // 32 for Gaudi

    ASSERT_EQ(numBlobs, 2);
    ASSERT_EQ(totalBlobsSizeInBytes, 2 * blobAlginment); // we expect that each blob will be filled with NOPs to reach blobAlginment byte size

    auto patchingBlobItr = std::find_if(commitedBlobs.begin(), commitedBlobs.end(), isPatchingLambda);
    auto exeBlobItr      = std::find_if(commitedBlobs.begin(), commitedBlobs.end(), isExeLambda);

    serializedBlob1 = &pBlobs[(*patchingBlobItr).index];
    serializedBlob2 = &pBlobs[(*exeBlobItr).index];

    ASSERT_TRUE(serializedBlob1->blob_type.requires_patching);
    ASSERT_FALSE(serializedBlob2->blob_type.requires_patching);

    ASSERT_EQ(serializedBlob1->size, blobAlginment); // we expect the blob to be round-up to 32 bytes using nop
    ASSERT_EQ(serializedBlob2->size, blobAlginment); // we expect the blob to be round-up to 32 bytes using nop

    uint8_t* readPtr = (uint8_t*)serializedBlob1->data;
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        ASSERT_EQ(*readPtr++, i);
    }
    readPtr = (uint8_t*)serializedBlob2->data;
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        ASSERT_EQ(*readPtr++, i * 2);
    }

}

void RecipeTestBase::blob_container_4_different_blobs(HabanaGraph *g)
{
    const uint64_t       blobSizeBytes = 32;
    RecipeBlobContainer  container(g);
    RecipeBlob*          blob1 = container.getPatchingBlob();
    RecipeBlob*          blob2 = container.getExecutionBlob();

    ASSERT_TRUE(blob1 != nullptr);
    ASSERT_TRUE(blob2 != nullptr);
    ASSERT_TRUE(blob1->isPatchingBlob());
    uint8_t* writePtr = blob1->reserveBytes(blobSizeBytes);
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        *writePtr++ = i * 1;
    }
    writePtr = blob2->reserveBytes(blobSizeBytes);
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        *writePtr++ = i * 2;
    }

    std::list<BlobCommitInfo> commitedBlobs = container.commitBlobs();

    auto patchingBlobItr = std::find_if(commitedBlobs.begin(), commitedBlobs.end(), isPatchingLambda);
    auto exeBlobItr      = std::find_if(commitedBlobs.begin(), commitedBlobs.end(), isExeLambda);

    ASSERT_EQ((*patchingBlobItr).index, 0);
    ASSERT_EQ((*exeBlobItr).index, 1);

    // Second pair of blobs
    blob1 = container.getPatchingBlob();
    blob2 = container.getExecutionBlob();
    ASSERT_TRUE(blob1 != nullptr);
    ASSERT_TRUE(blob2 != nullptr);
    writePtr = blob1->reserveBytes(blobSizeBytes);
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        *writePtr++ = i * 3;
    }
    writePtr = blob2->reserveBytes(blobSizeBytes);
    for (unsigned i = 0; i < blobSizeBytes; i++)
    {
        *writePtr++ = i * 4;
    }

    commitedBlobs = container.commitBlobs();

    patchingBlobItr = std::find_if(commitedBlobs.begin(), commitedBlobs.end(), isPatchingLambda);
    exeBlobItr      = std::find_if(commitedBlobs.begin(), commitedBlobs.end(), isExeLambda);

    ASSERT_EQ((*patchingBlobItr).index, 2);
    ASSERT_EQ((*exeBlobItr).index, 3);

    // Serialize
    uint64_t  numBlobs;
    blob_t*   pBlobs;
    uint64_t  totalBlobsSizeInBytes;
    blob_t*   serializedBlob;
    uint64_t* executionBlobBuffer;
    uint64_t* patchingBlobBuffer;
    uint32_t* workDistBlobBuffer;
    uint64_t  executionBlobBufferSize;
    uint64_t  patchingBlobBufferSize;
    uint64_t  workDistBlobBufferSize;

    RecipeAllocator recipeAlloc;

    container.serialize(&numBlobs,
                        &pBlobs,
                        &totalBlobsSizeInBytes,
                        &executionBlobBuffer,
                        &executionBlobBufferSize,
                        &patchingBlobBuffer,
                        &patchingBlobBufferSize,
                        &workDistBlobBuffer,
                        &workDistBlobBufferSize,
                        &recipeAlloc);

    ASSERT_EQ(numBlobs, 4);

    uint blobAlginment = g->getHALReader()->getCpDmaAlignment(); // 32 for Gaudi

    for (unsigned j = 0; j < numBlobs; j++)
    {
        serializedBlob = &pBlobs[j];
        ASSERT_EQ(serializedBlob->size, blobAlginment);
        uint8_t* readPtr = (uint8_t*)serializedBlob->data;
        for (unsigned i = 0; i < blobSizeBytes; i++)
        {
            ASSERT_EQ(*readPtr++, i * (j+1));
        }
    }

    if (LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN))
    {
        container.print();
    }
}

void RecipeTestBase::blob_container_blobs_chunks_test(HabanaGraph *g)
{
    // We'll define several blobs from each type and make sure that the blobs data
    // is correctly with and without the blobs chunk machanism.
    RecipeBlobContainer  container(g);

    static const uint64_t blobsNumToCreate        = 36;
    RecipeBlob* patchingBlobs[blobsNumToCreate]   = { nullptr };
    RecipeBlob* executionBlobs[blobsNumToCreate]  = { nullptr };
    uint64_t blobsSizesToCreate[blobsNumToCreate] = { 3648, 64,   64,   576,  64,   3648,
                                                      576,  64,   3648, 3648, 3648, 3648,
                                                      3648, 3648, 3648, 3648, 3648, 3648,
                                                      3648, 3648, 3648, 3648, 3648, 3648,
                                                      3648, 3648, 3648, 3648, 3648, 3648,
                                                      576,  3648, 3648, 3648, 3648, 3648 };


    uint8_t data = 1;
    for (unsigned i = 0; i < blobsNumToCreate; i++)
    {
        patchingBlobs[i] = container.getPatchingBlob();
        ASSERT_TRUE(patchingBlobs[i] != nullptr);
        uint8_t* writePtr = patchingBlobs[i]->reserveBytes(blobsSizesToCreate[i]);
        for (unsigned j = 0; j < blobsSizesToCreate[i]; j++)
        {
            *writePtr++ = data;
        }

        data++;
        executionBlobs[i] = container.getExecutionBlob();
        ASSERT_TRUE(executionBlobs[i] != nullptr);
        writePtr = executionBlobs[i]->reserveBytes(blobsSizesToCreate[i]);
        for (unsigned j = 0; j < blobsSizesToCreate[i]; j++)
        {
            *writePtr++ = data;
        }
        data++;
        std::list<BlobCommitInfo> commitedBlobs = container.commitBlobs();
    }

    // Serialize
    uint64_t  numBlobs;
    blob_t*   pBlobs;
    uint64_t  totalBlobsSizeInBytes;
    uint64_t* executionBlobBuffer;
    uint64_t* patchingBlobBuffer;
    uint32_t* workDistBlobBuffer;
    uint64_t  executionBlobBufferSize;
    uint64_t  patchingBlobBufferSize;
    uint64_t  workDistBlobBufferSize;

    RecipeAllocator recipeAlloc;

    container.serialize(&numBlobs,
                        &pBlobs,
                        &totalBlobsSizeInBytes,
                        &executionBlobBuffer,
                        &executionBlobBufferSize,
                        &patchingBlobBuffer,
                        &patchingBlobBufferSize,
                        &workDistBlobBuffer,
                        &workDistBlobBufferSize,
                        &recipeAlloc);

    ASSERT_EQ(numBlobs, 2 * blobsNumToCreate);

    blob_t* serializedBlob;
    unsigned k = 0;
    data = 1;

    for (unsigned j = 0; j < numBlobs; j++)
    {
        // Read from patching blob
        serializedBlob = &pBlobs[j++];
        uint8_t* readPtr = (uint8_t*)serializedBlob->data;
        for (unsigned i = 0; i < blobsSizesToCreate[k]; i++)
        {
            ASSERT_EQ(*readPtr++, data);
        }
        data++;
        // Read from execution blob
        serializedBlob = &pBlobs[j];
        readPtr = (uint8_t*)serializedBlob->data;
        for (unsigned i = 0; i < blobsSizesToCreate[k]; i++)
        {
            ASSERT_EQ(*readPtr++, data);
        }
        k++;
        data++;
    }

}
