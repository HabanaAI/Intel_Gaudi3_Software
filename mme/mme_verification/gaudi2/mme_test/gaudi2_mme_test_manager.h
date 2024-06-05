#include "mme_test_manager.h"

namespace gaudi2
{
class Gaudi2MmeTestManager : public MmeCommon::MmeTestManager
{
public:
    Gaudi2MmeTestManager() : MmeTestManager(MmeCommon::e_mme_Gaudi2) {};
    virtual ~Gaudi2MmeTestManager() = default;
protected:
    void makeMmeUser(unsigned mmeLimit) final;
    void
    makeDeviceHandler(MmeCommon::DeviceType devA, MmeCommon::DeviceType devB, std::vector<unsigned>& deviceIdxs) final;
    void makeSyncObjectManager(const uint64_t smBase, unsigned mmeLimit) override;
    std::shared_ptr<MmeCommon::MmeMemAccessChecker> createAccessChecker(unsigned euNr) override;
    bool verifyChipSpecificTestParams(nlohmann::json& testJson) override;
};
}  // namespace gaudi2