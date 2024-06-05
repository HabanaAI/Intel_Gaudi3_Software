#include "gaudi2_mme_test_manager.h"
#include "gaudi2/mme_user/gaudi2_mme_user.h"
#include "mme_verification/gaudi2/mme_user/gaudi2_sync_object_manager.h"
#include "mme_mem_access_checker_gaudi2.h"
#include "gaudi2_device_handler.h"

namespace gaudi2
{
void Gaudi2MmeTestManager::makeMmeUser(unsigned mmeLimit)
{
    m_mmeUser = std::make_unique<gaudi2::Gaudi2MmeUser>(mmeLimit);
}

void Gaudi2MmeTestManager::makeDeviceHandler(MmeCommon::DeviceType devA,
                                             MmeCommon::DeviceType devB,
                                             std::vector<unsigned>& deviceIdxs)
{
    m_devHandler = std::make_unique<gaudi2::Gaudi2DeviceHandler>(devA, devB, deviceIdxs);
}

void Gaudi2MmeTestManager::makeSyncObjectManager(const uint64_t smBase, unsigned mmeLimit)
{
    m_syncObjectManager = std::make_unique<gaudi2::Gaudi2SyncObjectManager>(smBase, mmeLimit);
}

std::shared_ptr<MmeCommon::MmeMemAccessChecker> Gaudi2MmeTestManager::createAccessChecker(unsigned euNr)
{
    return std::make_shared<MmeMemAccessCheckerGaudi2>(euNr);
}

bool Gaudi2MmeTestManager::verifyChipSpecificTestParams(nlohmann::json& testJson)
{
    return true;
}

}  // namespace gaudi2
