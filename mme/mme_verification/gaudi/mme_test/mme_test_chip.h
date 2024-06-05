#pragma once
#include <vector>
#include <string>
#include "mme_test_gen.h"
#include "gaudi_device_handler.h"
namespace gaudi
{
bool runChipTests(std::vector<MmeTestParams_t>* testsParams,
                  const std::string& dumpDir,
                  const std::string& lfsrDir,
                  const MmeCommon::DeviceType devTypeA,
                  const MmeCommon::DeviceType devTypeB,
                  const unsigned devAIdx,
                  const unsigned devBIdx,
                  bool verifMode,
                  bool gaudiM);

}  // namespace gaudi