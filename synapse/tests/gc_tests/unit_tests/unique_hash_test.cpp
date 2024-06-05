#include "gtest/gtest.h"
#include "recipe_blob.h"
#include "recipe_program.h"
#include "recipe_generator.h"
#include "recipe_allocator.h"
#include "unique_hash_test.h"

void UniqueHash::blob_equality(HabanaGraph *g)
{
    RecipeBlobContainer blobContainer(g);
    auto blob1 = new RecipeBlob(g);
    std::fill_n(blob1->reserveBytes(16), 16, 5);
    auto blob2 = new RecipeBlob(g);
    std::fill_n(blob2->reserveBytes(16), 16, 5);
    uint64_t i1;
    uint64_t i2;
    bool     isReused1;
    bool     isReused2;
    blobContainer.addBlob(blob1, i1, isReused1);
    blobContainer.addBlob(blob2, i2, isReused2);
    EXPECT_EQ(i1, i2);
    EXPECT_FALSE(isReused1);
    EXPECT_TRUE(isReused2);
}

void UniqueHash::blob_inequality(HabanaGraph *g)
{
    uint64_t idx;
    bool     isReused;
    RecipeBlobContainer blobContainer(g);
    auto blob1 = new RecipeBlob(g);
    std::fill_n(blob1->reserveBytes(8), 8, 0);
    blobContainer.addBlob(blob1, idx, isReused);
    // Change each bit once
    for (int i = 0; i < 8*8; ++i)
    {
        auto bytePosition = i / 8;
        auto bit = i % 8;

        auto blob2 = new RecipeBlob(g);
        auto data = blob2->reserveBytes(8);
        std::fill_n(data, 8, 0);
        data[bytePosition] ^= 1 << bit;
        blobContainer.addBlob(blob2, idx, isReused);
    }

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

    blobContainer.serialize(&numBlobs,
                            &pBlobs,
                            &totalBlobsSizeInBytes,
                            &executionBlobBuffer,
                            &executionBlobBufferSize,
                            &patchingBlobBuffer,
                            &patchingBlobBufferSize,
                            &workDistBlobBuffer,
                            &workDistBlobBufferSize,
                            &recipeAlloc);

    EXPECT_EQ(numBlobs, 8*8 + 1);

    // try to insert the first blob, see that the count doesn't change
    auto blob3 = new RecipeBlob(g);
    std::fill_n(blob3->reserveBytes(8), 8, 0);
    blobContainer.addBlob(blob3, idx, isReused);

    blobContainer.serialize(&numBlobs,
                            &pBlobs,
                            &totalBlobsSizeInBytes,
                            &executionBlobBuffer,
                            &executionBlobBufferSize,
                            &patchingBlobBuffer,
                            &patchingBlobBufferSize,
                            &workDistBlobBuffer,
                            &workDistBlobBufferSize,
                            &recipeAlloc);

    EXPECT_EQ(numBlobs, 8*8 + 1);
    EXPECT_TRUE(isReused);

}
