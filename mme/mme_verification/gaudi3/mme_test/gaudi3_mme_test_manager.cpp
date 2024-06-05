#include "gaudi3_mme_test_manager.h"
#include "include/mme_common/mme_common_enum.h"
#include "mme_user/gaudi3_mme_user.h"
#include "mme_verification/gaudi3/mme_user/gaudi3_sync_object_manager.h"
#include "mme_mem_access_checker_gaudi3.h"
#include "gaudi3_device_handler.h"

namespace gaudi3
{
void Gaudi3MmeTestManager::makeMmeUser(unsigned mmeLimit)
{
    m_mmeUser = std::make_unique<gaudi3::Gaudi3MmeUser>(mmeLimit);
}

void Gaudi3MmeTestManager::makeDeviceHandler(MmeCommon::DeviceType devA,
                                             MmeCommon::DeviceType devB,
                                             std::vector<unsigned>& deviceIdxs)
{
    m_devHandler = std::make_unique<gaudi3::Gaudi3DeviceHandler>(devA, devB, deviceIdxs);
}

void Gaudi3MmeTestManager::makeSyncObjectManager(const uint64_t smBase, unsigned mmeLimit)
{
    m_syncObjectManager = std::make_unique<gaudi3::Gaudi3SyncObjectManager>(smBase, mmeLimit);
}

std::shared_ptr<MmeCommon::MmeMemAccessChecker> Gaudi3MmeTestManager::createAccessChecker(unsigned euNr)
{
    return std::make_shared<MmeMemAccessCheckerGaudi3>();
}

bool Gaudi3MmeTestManager::verifyChipSpecificTestParams(nlohmann::json& testJson)
{
    if (testJson["cacheMode"] == true && m_devHandler->isRunOnChip())
    {
        for (auto& device : m_devHandler->getDriverDevices())
        {
            // TODO: verify in cache mode : SW-91115
        }
    }
    if (testJson["inTypeFloat"].get<MmeCommon::EMmeDataType>() == MmeCommon::EMmeDataType::e_type_fp32_ieee)
    {
        testJson["inTypeFloat"] = MmeCommon::EMmeDataType::e_type_fp32;
    }
    if (testJson["in2TypeFloat"].get<MmeCommon::EMmeDataType>() == MmeCommon::EMmeDataType::e_type_fp32_ieee)
    {
        testJson["in2TypeFloat"] = MmeCommon::EMmeDataType::e_type_fp32;
    }
    return true;
}
}  // namespace gaudi3
