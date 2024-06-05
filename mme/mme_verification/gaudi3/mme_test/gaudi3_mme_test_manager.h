#include "mme_test_manager.h"

namespace gaudi3
{
class Gaudi3MmeTestManager : public MmeCommon::MmeTestManager
{
public:
    Gaudi3MmeTestManager() : MmeTestManager(MmeCommon::e_mme_Gaudi3) {};

protected:
    void makeMmeUser(unsigned mmeLimit) final;
    void
    makeDeviceHandler(MmeCommon::DeviceType devA, MmeCommon::DeviceType devB, std::vector<unsigned>& deviceIdxs) final;
    void makeSyncObjectManager(const uint64_t smBase, unsigned mmeLimit) override;
    std::shared_ptr<MmeCommon::MmeMemAccessChecker> createAccessChecker(unsigned euNr) override;
    bool verifyChipSpecificTestParams(nlohmann::json& testJson) override;
};
}  // namespace gaudi3