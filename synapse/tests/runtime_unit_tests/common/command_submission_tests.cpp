#include <gtest/gtest.h>

#define protected public
#define private public

#include "runtime/qman/common/command_submission.hpp"
#include "synapse_common_types.h"

#include "drm/habanalabs_accel.h"

class UTCommandSubmissionTest : public ::testing::Test
{
public:
private:
};

TEST_F(UTCommandSubmissionTest, command_submission_too_many_chunks)
{
    CommandSubmission cmdSubmission;
    void*             pExecuteChunkArgs      = nullptr;
    uint32_t          executeChunksAmount    = 0;
    bool              isRequireExternalChunk = true;
    uint32_t          queueOffset            = 0;
    const StagedInfo* pStagedInfo            = nullptr;

    // A single external execution is required
    cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, 1, 0);

    // Add Execution PQ-Entries
    // Fill up all the reset with internals (Execution)
    for (unsigned i = 0; i < HL_MAX_JOBS_PER_CS - 1; i++)
    {
        cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_INTERNAL_EXECUTION, 0, 1, 0);
    }
    synStatus status = cmdSubmission.prepareForSubmission(pExecuteChunkArgs,
                                                          executeChunksAmount,
                                                          isRequireExternalChunk,
                                                          queueOffset,
                                                          pStagedInfo);
    ASSERT_EQ(status, synSuccess);

    // Remove the external (execution)
    cmdSubmission.clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    status = cmdSubmission.prepareForSubmission(pExecuteChunkArgs,
                                                executeChunksAmount,
                                                isRequireExternalChunk,
                                                queueOffset,
                                                pStagedInfo);
    ASSERT_EQ(status, synInvalidArgument);

    // Return the external and add another internal
    cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, 1, 0);
    cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_INTERNAL_EXECUTION, 0, 1, 0);
    status = cmdSubmission.prepareForSubmission(pExecuteChunkArgs,
                                                executeChunksAmount,
                                                isRequireExternalChunk,
                                                queueOffset,
                                                pStagedInfo);
    ASSERT_EQ(status, synCommandSubmissionFailure);

    // Repeat with execute PQ-Entries
    cmdSubmission.clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    cmdSubmission.clearPrimeQueueEntries(PQ_ENTRY_TYPE_INTERNAL_EXECUTION);

    // A single external is required
    cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, 1, 0);

    // Fill up all the reset with internals (Execute)
    for (unsigned i = 0; i < HL_MAX_JOBS_PER_CS - 1; i++)
    {
        cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_INTERNAL_EXECUTION, 0, 1, 0);
    }
    status = cmdSubmission.prepareForSubmission(pExecuteChunkArgs,
                                                executeChunksAmount,
                                                isRequireExternalChunk,
                                                queueOffset,
                                                pStagedInfo);
    ASSERT_EQ(status, synSuccess);

    // Remove the external (execution)
    cmdSubmission.clearPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    status = cmdSubmission.prepareForSubmission(pExecuteChunkArgs,
                                                executeChunksAmount,
                                                isRequireExternalChunk,
                                                queueOffset,
                                                pStagedInfo);
    ASSERT_EQ(status, synInvalidArgument);

    // Return the external and add another internal
    cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, 0, 1, 0);
    cmdSubmission.addPrimeQueueEntry(PQ_ENTRY_TYPE_INTERNAL_EXECUTION, 0, 1, 0);
    status = cmdSubmission.prepareForSubmission(pExecuteChunkArgs,
                                                executeChunksAmount,
                                                isRequireExternalChunk,
                                                queueOffset,
                                                pStagedInfo);
    ASSERT_EQ(status, synCommandSubmissionFailure);
}