#include "mme_test.h"
#include "mme_test/gaudi3_mme_test_manager.h"
#include "include/mme_common/mme_common_enum.h"

int main(int argc, char** argv)
{
    std::unique_ptr<MmeCommon::MmeTestManager> pMmeTestManager = std::make_unique<gaudi3::Gaudi3MmeTestManager>();
    MmeCommon::MMETest test(std::move(pMmeTestManager), MmeCommon::e_mme_Gaudi3);
    return test.run(argc, argv);
}
