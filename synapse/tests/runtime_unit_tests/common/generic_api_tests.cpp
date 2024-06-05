#include "synapse_api.h"
#include "synapse_common_types.hpp"

#include <gtest/gtest.h>

TEST(SynAPIUnitTests, status_description_api_verification)
{
    synStatus status = synSuccess;
    size_t    len    = STATUS_DESCRIPTION_MAX_SIZE;
    char      statusDescription[len];

    size_t maxSize = 0;
    for (uint32_t i = 0; i <= synStatusLast; i++)
    {
        status = synStatusGetBriefDescription((synStatus) i, statusDescription, len);
        ASSERT_EQ(status, synSuccess) << "Failed to get status description for " << i;

        maxSize = std::max(maxSize, strlen(statusDescription));
    }
    ASSERT_LT(maxSize, STATUS_DESCRIPTION_MAX_SIZE) << "Status description max-size is greated than "
                                                    << STATUS_DESCRIPTION_MAX_SIZE;

    status = synStatusGetBriefDescription(synSuccess, nullptr, len);
    ASSERT_EQ(status, synInvalidArgument) << "Unexpectedly success while providing nullptr chars' bufffer";
}