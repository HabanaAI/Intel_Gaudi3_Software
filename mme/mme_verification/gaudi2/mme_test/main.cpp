#include "mme_test.h"
#include "mme_test/gaudi2_mme_test_manager.h"
#include "include/mme_common/mme_common_enum.h"

int main(int argc, char** argv)
{
    std::unique_ptr<MmeCommon::MmeTestManager> pMmeTestManager = std::make_unique<gaudi2::Gaudi2MmeTestManager>();
    MmeCommon::MMETest test(std::move(pMmeTestManager), MmeCommon::e_mme_Gaudi2);
    return test.run(argc, argv);
}
