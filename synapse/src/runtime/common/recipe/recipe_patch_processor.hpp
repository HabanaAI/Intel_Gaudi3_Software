#pragma once

#include <cstdint>
#include <array>
#include <map>
#include <vector>
#include <unordered_set>

#include "synapse_common_types.h"
#include "synapse_api_types.h"
#include "settable.h"
#include "runtime/common/recipe/patching/define.hpp"

struct basicRecipeInfo;
struct RecipeTensorsInfo;
class RecipeStaticInfo;
class DevMemoryAllocInterface;
struct WorkSpacesInformation;
struct DeviceAgnosticRecipeInfo;

namespace patching
{
class HostAddressPatchingInformation;
}
struct SectionSizeTensor
{
    uint64_t sectionSize;
    uint64_t lastTensorRecipeidx;  // invalid for const section
    bool     isConstSection;
};

struct ValidSectionAddresses
{
    ValidSectionAddresses()
    {
        lowestValidAddress  = 0;
        highestValidAddress = 0;
    }
    ValidSectionAddresses(uint64_t low, uint64_t high) : lowestValidAddress(low), highestValidAddress(high) {}

    uint64_t lowestValidAddress;
    uint64_t highestValidAddress;
};

using namespace patching;

struct SectionsInformation
{
    patching::SectionToSectionType sectionToSectionType;
    std::vector<SectionSizeTensor> sectionsInfo;
    Settable<uint64_t>             maxSectionId;
    Settable<uint64_t>             numSections;
};

struct WorkSpacesInformation
{
    WorkSpacesInformation();

    uint64_t scratchPadAddress;
    uint64_t programCodeAddress;
    uint64_t programDataAddress;
    uint64_t assertAsyncMappedAddress;
    bool     programCodeInCache;
    bool     programDataInCache;
};

typedef SmallVector<SectionSizeTensor, 8> SectionSizeInfoVec;

class RecipePatchProcessor
{
    struct ValidateSectionInfo
    {
        uint64_t sectionId;
        uint64_t sectionSize;
        bool     isConstSection;
    };

    using SortedSectionsAddressInfo = std::map<uint64_t, ValidateSectionInfo>;  // [address, [section_id,size]]

    using SectionsAddressDB = std::vector<uint64_t>;

    // A vector of booleans per tensor-index (of a given tensor-type)
    using TensorsValidationInfoDB = std::vector<bool>;

public:
    static synStatus process(const basicRecipeInfo&                    rBasicRecipeInfo,
                             const RecipeTensorsInfo&                  rRecipeTensorsInfo,
                             const synLaunchTensorInfoExt*             enqueueTensorsInfo,
                             uint32_t                                  enqueueTensorsInfoAmount,
                             uint32_t                                  enqueueFlags,
                             const WorkSpacesInformation&              rWorkSpacesInformation,
                             patching::HostAddressPatchingInformation& rPatchInfo,
                             DevMemoryAllocInterface&                  rDevMemAlloc,
                             std::vector<uint32_t>*                    tensorIdx2userIdx,
                             bool                                      isInitAndCompletionRequired,
                             bool                                      shouldResolveTensorsIndices,
                             const ValidSectionAddresses*              pValidSectionAddresses);

    // validate the sections new base offset for given tensors.
    static synStatus validateSectionsInfo(const basicRecipeInfo&         rBasicRecipeInfo,
                                          const RecipeTensorsInfo&       rRecipeInfo,
                                          const uint32_t                 launchTensorsAmount,
                                          const synLaunchTensorInfoExt*  launchTensorsInfo,
                                          uint32_t                       flags,
                                          const WorkSpacesInformation&   workspaceInfo,
                                          const SectionSizeInfoVec&      sectionsSizeInfo,
                                          DevMemoryAllocInterface&       rDevMemAlloc);

    static bool resolveTensorsIndices(std::vector<uint32_t>*        tensorIdx2userIdx,
                                      const basicRecipeInfo&        rBasicRecipeInfo,
                                      const uint32_t                launchTensorsAmount,
                                      const synLaunchTensorInfoExt* launchTensorsInfo);

    static synStatus resolveSectionsAddresses(const basicRecipeInfo&                    rBasicRecipeInfo,
                                              const uint32_t                            launchTensorsAmount,
                                              const synLaunchTensorInfoExt*             launchTensorsInfo,
                                              patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
                                              const SectionToSectionType*               sectionToSectionType,
                                              DevMemoryAllocInterface&                  rDevMemAlloc,
                                              const WorkSpacesInformation&              rWorkSpacesInformation,
                                              const RecipeTensorsInfo&                  rRecipeTensorInfo,
                                              const ValidSectionAddresses*              pValidSectionAddresses);

private:
    // brief calculate sections new base offset for given tensors.
    static bool _setWorkspacesSectionAddress(const basicRecipeInfo&                    rBasicRecipeInfo,
                                             const WorkSpacesInformation&              workspaceInfo,
                                             patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
                                             const SectionToSectionType*               pSectionToSectionType = nullptr);

    static synStatus _checkTensorAddresses(const basicRecipeInfo&                    rBasicRecipeInfo,
                                           const synLaunchTensorInfoExt*             launchTensorsInfo,
                                           DevMemoryAllocInterface&                  rDevMemAlloc,
                                           uint32_t                                  tensorIdx,
                                           uint64_t                                  tensorId,
                                           const RecipeTensorsInfo&                  rRecipeTensorInfo,
                                           const ValidSectionAddresses*              pValidSectionAddresses);

    static bool _checkWorkspaceAddresses(const basicRecipeInfo&                    rBasicRecipeInfo,
                                         const WorkSpacesInformation&              rWorkSpacesInformation,
                                         const ValidSectionAddresses*              pValidSectionAddresses);

    static synStatus _resolveSectionsAddressesAndTensorsIndices(
        const basicRecipeInfo&                    rBasicRecipeInfo,
        const RecipeTensorsInfo&                  recipeTensorInfo,
        const uint32_t                            launchTensorsAmount,
        const synLaunchTensorInfoExt*             launchTensorsInfo,
        uint32_t                                  flags,
        patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
        std::vector<uint32_t>*                    tensorIdx2userIdx,
        const SectionToSectionType*               sectionToSectionType,
        DevMemoryAllocInterface&                  rDevMemAlloc,
        bool                                      shouldResolveTensorsIndices,
        const WorkSpacesInformation&              rWorkSpacesInformation,
        const ValidSectionAddresses*              pValidSectionAddresses);


    static synStatus _resolveBothSectionsAddressesAndTensorsIndices(
        const basicRecipeInfo&                    rBasicRecipeInfo,
        const uint32_t                            launchTensorsAmount,
        const synLaunchTensorInfoExt*             launchTensorsInfo,
        std::vector<uint32_t>*                    tensorIdx2userIdx,
        patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
        const SectionToSectionType*               sectionToSectionType,
        DevMemoryAllocInterface&                  rDevMemAlloc,
        const WorkSpacesInformation&              rWorkSpacesInformation,
        const RecipeTensorsInfo&                  rRecipeTensorInfo,
        const ValidSectionAddresses*              pValidSectionAddresses);


    static tensor_info_t::ETensorType userTypeToRecipeType(synTensorType type);

    static bool _addRecipeReservedSectionsWorkspaceAddresses(SortedSectionsAddressInfo&   sortedSections,
                                                             const basicRecipeInfo&       rBasicRecipeInfo,
                                                             const WorkSpacesInformation& workspaceInfo);

    static bool _validateTensorType(bool                 isDsd,
                                    uint64_t             tensorInfoIndex,
                                    const char*&         tensorName,
                                    const synTensorType& launchTensorType,
                                    const synTensorType& expectedType);

    static bool _shouldHandleValidTensor(TensorsValidationInfoDB* tensorsValidationInfo,
                                         uint64_t*                tensorCntDB,
                                         uint64_t                 tensorInfoIndex,
                                         const char*&             tensorName,
                                         const synTensorType&     launchTensorType);

    static bool _addTensorSectionAddressToDB(SectionsAddressDB& sectionAddresses,
                                             uint64_t           currSectionIdx,
                                             uint64_t           currSectionAddr);

    static bool _insertSectionSize(SortedSectionsAddressInfo&    sortedSections,
                                   uint64_t                      currSectionIdx,
                                   uint64_t                      tensorInfoIndex,
                                   uint64_t                      currSectionAddr,
                                   const SectionSizeInfoVec&     sectionsSizeInfo,
                                   const synLaunchTensorInfoExt& launchTensorInfo,
                                   const synTensorType&          launchTensorType,
                                   const persist_tensor_info_t*  recipePersistentTensors,
                                   uint64_t                      numOfPersistentTensors);

    static bool _markConstZeroSizeTensors(const std::unordered_set<uint64_t>& constZeroSizeTensors,
                                          TensorsValidationInfoDB*            tensorGiven,
                                          uint64_t*                           tensorCnt);

    static bool _validateAllTensorsGiven(uint16_t                       tensorTypeIndex,
                                         uint64_t                       tensorCnt,
                                         uint64_t                       expectedNumTensors,
                                         const TensorsValidationInfoDB& tensorValidationInfo,
                                         const persist_tensor_info_t*   persistentTensorsDB,
                                         const shape_tensor_info_t*     shapeTensorsDB,
                                         uint64_t                       numOfPersistentTensors,
                                         uint64_t                       numOfShapeTensors);

    static bool _checkSectionsOverlap(const SortedSectionsAddressInfo& sortedSections);

    static void _logLaunchTensorsInfo(const SortedSectionsAddressInfo& sortedSections,
                                      const synLaunchTensorInfoExt*    launchTensorsInfo,
                                      const persist_tensor_info_t*     persistentTensorsDB,
                                      uint64_t                         numOfPersistentTensors,
                                      uint32_t                         launchTensorsAmount,
                                      uint64_t                         maxSectionId);

    static bool _isOutOfRange(uint64_t baseAddr, uint64_t size, const ValidSectionAddresses* pValidSectionAddr);

    static void _retrieveTensorInfo(tensor_info_t::ETensorType& tensorRecipeType,
                                    synTensorType&              tensorType,
                                    uint64_t&                   tensorIndex,
                                    uint64_t                    tensorId);

    static bool _resolveSingleTensorIdToTensorIndex(tensor_info_t::ETensorType& tensorRecipeType,
                                                    uint64_t&                   tensorIndex,
                                                    uint64_t                    numberShapeTensors,
                                                    uint64_t                    numberPersistTensors);

    static synStatus _resolveSingleTensorSectionsAddresses(const basicRecipeInfo&                    rBasicRecipeInfo,
                                                           const synLaunchTensorInfoExt*             pLaunchTensorsInfo,
                                                           patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
                                                           const SectionToSectionType*               sectionToSectionType,
                                                           const RecipeTensorsInfo&                  rRecipeTensorInfo,
                                                           DevMemoryAllocInterface&                  rDevMemAlloc,
                                                           uint64_t                                  tensorIndex);
};