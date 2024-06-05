#pragma once

#include "synapse_common_types.h"
#include "syn_exception.hpp"
#include <vector>
#include <memory>
#include <sstream>

#define SYN_FILENAME    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define SYN_UNLIKELY(x) __builtin_expect((x), 0)

namespace syn
{
constexpr uint32_t maxStringLength = 1024;

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch-enum"
inline const char* toString(synStatus status)
{
#define SYN_STR(x)                                                                                                     \
    case x:                                                                                                            \
        return #x;

    switch (status)
    {
        SYN_STR(synSuccess);
        SYN_STR(synInvalidArgument);
        SYN_STR(synCbFull);
        SYN_STR(synOutOfHostMemory);
        SYN_STR(synOutOfDeviceMemory);
        SYN_STR(synObjectAlreadyInitialized);
        SYN_STR(synObjectNotInitialized);
        SYN_STR(synCommandSubmissionFailure);
        SYN_STR(synNoDeviceFound);
        SYN_STR(synDeviceTypeMismatch);
        SYN_STR(synFailedToInitializeCb);
        SYN_STR(synFailedToFreeCb);
        SYN_STR(synFailedToMapCb);
        SYN_STR(synFailedToUnmapCb);
        SYN_STR(synFailedToAllocateDeviceMemory);
        SYN_STR(synFailedToFreeDeviceMemory);
        SYN_STR(synFailedNotEnoughDevicesFound);
        SYN_STR(synOutOfResources);
        SYN_STR(synDeviceReset);
        SYN_STR(synUnsupported);
        SYN_STR(synWrongParamsFile);
        SYN_STR(synDeviceAlreadyAcquired);
        SYN_STR(synNameIsAlreadyUsed);
        SYN_STR(synBusy);
        SYN_STR(synAllResourcesTaken);
        SYN_STR(synUnavailable);
        SYN_STR(synInvalidTensorDimensions);
        SYN_STR(synFail);
        SYN_STR(synUninitialized);
        SYN_STR(synAlreadyInitialized);
        SYN_STR(synFailedSectionValidation);
        SYN_STR(synSynapseTerminated);
        SYN_STR(synAssertAsync);
        SYN_STR(synInvalidEventHandle);
        SYN_STR(synMappingNotFound);
        SYN_STR(synFailedDynamicPatching);
        SYN_STR(synFailedStaticPatching);
        SYN_STR(synFailedToSubmitWorkload);
        SYN_STR(synInvalidSectionsDefinition);
        SYN_STR(synInvalidTensorProperties);
        SYN_STR(synFailHccl);
        SYN_STR(synFailedToCollectTime);
        SYN_STR(synTimeout);
        SYN_STR(synResourceBadUsage);
    }
#undef SYN_STR
    return "<Unknown Status>";
}
#pragma GCC diagnostic pop

#define SYN_CHECK(cmd_)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        synStatus status_ = (cmd_);                                                                                    \
        if (SYN_UNLIKELY(status_ != synSuccess))                                                                       \
        {                                                                                                              \
            std::stringstream msg;                                                                                     \
            msg << toString(status_) << ", in file: " << SYN_FILENAME << " (" << __LINE__ << ")"                       \
                << ", function: " << __func__ << ", after: " << #cmd_;                                                 \
            throw Exception(status_, msg.str());                                                                       \
        }                                                                                                              \
    } while (false)

#define SYN_THROW_IF(cond_, status_)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((cond_))                                                                                                   \
        {                                                                                                              \
            SYN_CHECK(status_);                                                                                        \
        }                                                                                                              \
    } while (false)

inline std::vector<const char*> toConstChar(const std::vector<std::string>& list, bool emptyToNull)
{
    std::vector<const char*> ret;
    ret.reserve(list.size());
    for (const auto& s : list)
    {
        ret.push_back(s.empty() && emptyToNull ? nullptr : s.c_str());
    }
    return ret;
}

struct TensorMetadata
{
    std::string           name;
    std::vector<TSize>    shape;
    uint64_t              offsetInSection;
    uint32_t              elementType;
    uint32_t              sectionId;
    bool                  isInput;
};
}  // namespace syn