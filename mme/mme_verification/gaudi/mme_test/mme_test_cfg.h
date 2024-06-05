#pragma once
#include <string>
#include <vector>
#include "mme_test_gen.h"

void cfgFile2Tests(const std::string& fileName, unsigned seed, std::vector<MmeTestParams_t> *testsParams);
std::string test2text(const MmeTestParams_t *tp);

