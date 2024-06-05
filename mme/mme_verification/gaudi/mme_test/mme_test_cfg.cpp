#include "mme_test_gen.h"
#include "mme_assert.h"
#include <fstream>
#include <iostream>
#include <list>
#include <regex>
#include <sstream>
#include <string.h>
#include <vector>

#define REGEX(s)   std::regex(s)
#define REGEX_I(s) std::regex(s, std::regex::ECMAScript | std::regex::icase)

#if __cplusplus >= 201103L &&                                                                                          \
    (!defined(__GLIBCXX__) || (__cplusplus >= 201402L) ||                                                              \
     (defined(_GLIBCXX_REGEX_DFS_QUANTIFIERS_LIMIT) || defined(_GLIBCXX_REGEX_STATE_LIMIT) ||                          \
      (defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE > 4)))
#define HAVE_WORKING_REGEX 1
#else
#define HAVE_WORKING_REGEX 0
#endif

#ifndef WIN32
#include <strings.h>
#define stricmp strcasecmp
#endif

using namespace MmeCommon;

struct Cfg
{
    std::string name;
    std::vector<std::string> val;
};

static void
getConfigLines(const std::string& fileName, std::vector<Cfg>* globalCfgs, std::vector<std::vector<Cfg>>* testsCfgs)
{
#if HAVE_WORKING_REGEX
    std::ifstream file(fileName.c_str());
    if (!file.is_open())
    {
        std::cerr << "ERROR: Failed to open input file (" << fileName << ")\n";
        exit(1);
    }

    bool global;
    std::smatch matches;
    std::string line;
    while (std::getline(file, line))
    {
        line = std::regex_replace(line, REGEX("//.*"), "");
        line = std::regex_replace(line, REGEX("\\s"), "");
        if (!line.size()) continue;

        if (std::regex_search(line, REGEX_I("^global:")))
        {
            global = true;
            line = std::regex_replace(line, REGEX_I("^global:"), "");
        }
        else
        {
            global = false;
            if (std::regex_search(line, REGEX_I("^testName=")))
            {
                testsCfgs->push_back(std::vector<Cfg>());
            }
        }

        if (std::regex_search(line, matches, REGEX_I("^([^=]+)=(.+)")))
        {
            std::vector<Cfg>& cfgList = global ? *globalCfgs : (*testsCfgs).back();
            cfgList.push_back(Cfg());
            cfgList.back().name = matches[1].str();

            std::string val = matches[2].str();
            while (std::regex_search(val, matches, REGEX_I("^([^,]+),?(.*)")))
            {
                cfgList.back().val.push_back(matches[1].str());
                val = matches[2].str();
            }
        }
    }
#else
    std::cerr << "ERROR: Compiler doesn't support regex - mme tests is not usable\n";
    exit(1);
#endif  // HAVE_WORKING_REGEX
}

static int64_t parseInt(const std::string& str)
{
    int64_t ret = 0;
#if HAVE_WORKING_REGEX
    if (std::regex_search(str, REGEX_I("^0x")))
    {
        ret = std::stoll(str, 0, 16);
    }
    else
    {
        ret = std::stoll(str);
    }
#endif  // HAVE_WORKING_REGEX

    return ret;
}

static void parseGlobalParams(const std::vector<Cfg>* globalCfgs, MmeTestParams_t* params)
{
    for (auto& cfg : *globalCfgs)
    {
        const char* name = cfg.name.c_str();
        if (cfg.val.size() != 1)
        {
            std::cerr << "ERROR: global attribute " << name << " must have exactly one value.\n";
            exit(1);
        }

        if (!stricmp(name, "sramBase"))
        {
            params->sramBase = parseInt(cfg.val[0]);
        }
        else if (!stricmp(name, "sramSize"))
        {
            params->sramSize = parseInt(cfg.val[0]);
        }
        else if (!stricmp(name, "hbmBase"))
        {
            params->hbmBase = parseInt(cfg.val[0]);
        }
        else if (!stricmp(name, "smBase"))
        {
            params->smBase = parseInt(cfg.val[0]);
        }
        else if (!stricmp(name, "multiplTests"))
        {
            params->multipleTests = (cfg.val[0] == "1");
        }
        else if (!stricmp(name, "fp"))
        {
            params->fp = (cfg.val[0] == "1");
        }
        else if (!stricmp(name, "shuffle"))
        {
            params->shuffle = (cfg.val[0] == "1");
        }
        else if (!stricmp(name, "programInSram"))
        {
            if (cfg.val[0] == "random")
            {
                params->programInSram = ((rand() & 0x1) == 1);
            }
            else if (cfg.val[0] == "1")
            {
                params->programInSram = true;
            }
            else if (cfg.val[0] == "0")
            {
                params->programInSram = false;
            }
            else
            {
                std::cerr << "ERROR: unknown option for programInSram: " << cfg.val[0] << "\n";
                exit(1);
            }
        }
        else if (!stricmp(name, "pole"))
        {
            params->pole = (cfg.val[0] == "north") ? NORTH_POLE : SOUTH_POLE;
        }
        else
        {
            std::cerr << "ERROR: unknown configuration " << name << ".\n";
            exit(1);
        }
    }
}

static const std::pair<const char*, EMmePattern> c_convStrategies[] = {
    {"ksf", e_mme_z_reduction_ksf},
    {"skf", e_mme_z_reduction_skf},
};

static const std::pair<const char*, EMmePattern> c_bgemmStrategies[] = {
    {"kcf", e_mme_sp_reduction_kcf},
    {"ckf", e_mme_sp_reduction_ckf},
    {"fkc", e_mme_sp_reduction_fkc},
    {"fck", e_mme_sp_reduction_fck},
};

static const std::pair<const char*, EMmePattern> c_dedwStrategies[] = {
    {"kfc", e_mme_sp_reduction_kfc},
    {"fkc", e_mme_sp_reduction_fkc},
    {"fck", e_mme_sp_reduction_fck},
    {"cfk", e_mme_sp_reduction_cfk},
    {"kcf", e_mme_sp_reduction_kcf},
    {"ckf", e_mme_sp_reduction_ckf},
};

static const std::pair<const char*, EConvTestOp> c_op[] = {
    {"fwd", E_CONV_TEST_FWD},
    {"dedx", E_CONV_TEST_DEDX},
    {"dedw", E_CONV_TEST_DEDW},
    {"bgemm_ab", E_CONV_TEST_AB},
    {"bgemm_abt", E_CONV_TEST_ABT},
    {"bgemm_atb", E_CONV_TEST_ATB},
    {"bgemm_atbt", E_CONV_TEST_ATBT},
};

static const std::pair<const char*, EMmeGeometry> c_geometry[] = {{"4w1h", e_mme_geometry_4wx1h},
                                                                  {"2w2h", e_mme_geometry_2wx2h},
                                                                  {"1w4h", e_mme_geometry_1wx4h}};

static const std::pair<const char*, RoundingMode> c_roundingModes[] = {{"rn", RoundToNearest},
                                                                       {"rz", RoundToZero},
                                                                       {"ru", RoundUp},
                                                                       {"rd", RoundDown}};

static void fixParamConflicts(MmeTestParams_t* params)
{
    if ((params->op == E_CONV_TEST_DEDX) || (params->op == E_CONV_TEST_DEDW))
    {
        params->conv.relu = false;
    }
}

static bool parseTestParams(const std::vector<Cfg>* testCfgs, const std::vector<int>* counters, MmeTestParams_t* params)
{
    // defaults
    params->setId = true;
    params->repeats = 1;
    params->sbReuseInStripes = 0;
    params->memsetOutput = 0;
    params->adjustFloatShapes = true;
    params->scaledRandomValues = false;
    params->incDec = false;
    params->loop = 0;
    params->memsetVoidPixels = false;
    params->stochsaticConversion = false;
    params->unrollEn = false;
    params->dedwAsBgemmEn = false;
    params->recurringMisalignmentOptEn = false;

    MME_ASSERT(counters->size() == testCfgs->size(), "counter size should be the same as test configs");
    for (int i = 0; i < testCfgs->size(); i++)
    {
        auto& cfg = (*testCfgs)[i];
        auto idx = (*counters)[i];
        auto& val = cfg.val[idx];
        const char* val_str = val.c_str();
        bool random = (!stricmp(val_str, "random"));

        const char* name = cfg.name.c_str();
        if (!stricmp(name, "testName"))
        {
            if (cfg.val.size() != 1)
            {
                std::cerr << "ERROR: attribute " << name << " must have exactly one value.\n";
                exit(1);
            }
            MME_ASSERT(strlen(val_str) <= sizeof(params->name) - 1,
                      "val_str should have the same size as attribute name");
            strcpy(params->name, val_str);
        }

        else if (!stricmp(name, "operation"))
        {
            if (random)
            {
                val_str = c_op[rand() % (sizeof(c_op) / sizeof(c_op[0]))].first;
            }
            bool found = false;
            for (int j = 0; j < (sizeof(c_op) / sizeof(c_op[0])); j++)
            {
                if (!stricmp(val_str, c_op[j].first))
                {
                    params->op = c_op[j].second;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                std::cerr << "ERROR: " << val_str << " is not a valid configuration for attribute " << name << "."
                          << std::endl;
                exit(1);
            }
        }

        else if (!stricmp(name, "reluEn"))
        {
            params->conv.relu = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "adjustFloatShapes"))
        {
            params->adjustFloatShapes = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "lowerEn"))
        {
            params->lower = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "sbReuse"))
        {
            params->sbReuse = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "unrollEn"))
        {
            params->unrollEn = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "dedwAsBgemmEn"))
        {
            params->dedwAsBgemmEn = random ? ((rand() & 0x1) == 1) : val == "1";
        }
        else if (!stricmp(name, "recurringMisalignmentOptEn"))
        {
            params->recurringMisalignmentOptEn = random ? ((rand() & 0x1) == 1) : val == "1";
        }
        else if (!stricmp(name, "sbReuseInStripes"))
        {
            params->sbReuseInStripes = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "signalPartial"))
        {
            params->signalPartial = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "memsetVoidPixels"))
        {
            params->memsetVoidPixels = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "repeats"))
        {
            params->repeats = parseInt(val);
        }

        else if (!stricmp(name, "xSizes"))
        {
            for (int j = 0; j < cfg.val.size(); j++)
            {
                params->inputShape[j] = parseInt(cfg.val[j]);
            }
        }

        else if (!stricmp(name, "ySizes"))
        {
            for (int j = 0; j < cfg.val.size(); j++)
            {
                params->outputShape[j] = parseInt(cfg.val[j]);
            }
            params->ioDim = cfg.val.size();
        }

        else if (!stricmp(name, "wSizes"))
        {
            for (int j = 0; j < cfg.val.size(); j++)
            {
                params->weightsShape[j] = parseInt(cfg.val[j]);
            }
            params->conv.dim = cfg.val.size() - 2;
        }

        else if (!stricmp(name, "dilation"))
        {
            for (int j = 0; j < cfg.val.size(); j++)
            {
                params->conv.dilation[j] = parseInt(cfg.val[j]);
            }
        }

        else if (!stricmp(name, "strides"))
        {
            for (int j = 0; j < cfg.val.size(); j++)
            {
                params->conv.convStride[j] = parseInt(cfg.val[j]);
            }
        }

        else if (!stricmp(name, "padding"))
        {
            for (int j = 0; j < cfg.val.size(); j++)
            {
                params->conv.padding[j] = parseInt(cfg.val[j]);
            }
        }

        else if (!stricmp(name, "inType"))
        {
            params->inputType = random                          ? (rand() & 0x1) ? e_type_bf16 : e_type_fp32
                                : (!stricmp(val_str, "bfloat")) ? e_type_bf16
                                                                : e_type_fp32;
        }

        else if (!stricmp(name, "outType"))
        {
            params->outputType = random                          ? (rand() & 0x1) ? e_type_bf16 : e_type_fp32
                                 : (!stricmp(val_str, "bfloat")) ? e_type_bf16
                                                                 : e_type_fp32;
        }

        else if (!stricmp(name, "paddingValue"))
        {
            if (params->fp)
            {
                float tmp = atof(val_str);
                params->conv.paddingValue.f32 = *(int32_t*) &tmp;
            }
            else
            {
                params->conv.paddingValue.f32 = parseInt(val);
            }

            if (params->inputType == e_type_bf16)
            {
                params->conv.paddingValue.f32 >>= 16;
            }
        }

        else if (!stricmp(name, "convPattern"))
        {
            if (params->op == E_CONV_TEST_FWD || params->op == E_CONV_TEST_DEDX)
            {
                if (random)
                {
                    val_str = c_convStrategies[rand() % (sizeof(c_convStrategies) / sizeof(c_convStrategies[0]))].first;
                }
                bool found = false;
                for (int j = 0; j < (sizeof(c_convStrategies) / sizeof(c_convStrategies[0])); j++)
                {
                    if (!stricmp(val_str, c_convStrategies[j].first))
                    {
                        params->pattern = c_convStrategies[j].second;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    std::cerr << "ERROR: " << val_str << " is not a valid configuration for attribute " << name << "."
                              << std::endl;
                    exit(1);
                }
            }
            else
            {
                if (idx) return false;
            }
        }

        else if (!stricmp(name, "bgemmPattern"))
        {
            if ((params->op == E_CONV_TEST_AB) || (params->op == E_CONV_TEST_ABT) || (params->op == E_CONV_TEST_ATB) ||
                (params->op == E_CONV_TEST_ATBT))
            {
                if (random)
                {
                    val_str =
                        c_bgemmStrategies[rand() % (sizeof(c_bgemmStrategies) / sizeof(c_bgemmStrategies[0]))].first;
                }
                bool found = false;
                for (int j = 0; j < (sizeof(c_bgemmStrategies) / sizeof(c_bgemmStrategies[0])); j++)
                {
                    if (!stricmp(val_str, c_bgemmStrategies[j].first))
                    {
                        params->pattern = c_bgemmStrategies[j].second;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    std::cerr << "ERROR: " << val_str << " is not a valid configuration for attribute " << name << "."
                              << std::endl;
                    exit(1);
                }
            }
            else
            {
                if (idx) return false;
            }
        }

        else if (!stricmp(name, "dedwPattern"))
        {
            if (params->op == E_CONV_TEST_DEDW)
            {
                if (random)
                {
                    val_str = c_dedwStrategies[rand() % (sizeof(c_dedwStrategies) / sizeof(c_dedwStrategies[0]))].first;
                }
                bool found = false;
                for (int j = 0; j < (sizeof(c_dedwStrategies) / sizeof(c_dedwStrategies[0])); j++)
                {
                    if (!stricmp(val_str, c_dedwStrategies[j].first))
                    {
                        params->pattern = c_dedwStrategies[j].second;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    std::cerr << "ERROR: " << val_str << " is not a valid configuration for attribute " << name << "."
                              << std::endl;
                    exit(1);
                }
            }
            else
            {
                if (idx) return false;
            }
        }

        else if (!stricmp(name, "geometry"))
        {
            if (random)
            {
                val_str = c_geometry[rand() % (sizeof(c_geometry) / sizeof(c_geometry[0]))].first;
            }
            bool found = false;
            for (int j = 0; j < (sizeof(c_geometry) / sizeof(c_geometry[0])); j++)
            {
                if (!stricmp(val_str, c_geometry[j].first))
                {
                    params->geometry = c_geometry[j].second;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                std::cerr << "ERROR: " << val_str << " is not a valid configuration for attribute " << name << "."
                          << std::endl;
                exit(1);
            }
        }

        else if (!stricmp(name, "roundingMode"))
        {
            if (random)
            {
                val_str = c_roundingModes[rand() % (sizeof(c_roundingModes) / sizeof(c_roundingModes[0]))].first;
            }
            bool found = false;
            for (int j = 0; j < (sizeof(c_roundingModes) / sizeof(c_roundingModes[0])); j++)
            {
                if (!stricmp(val_str, c_roundingModes[j].first))
                {
                    params->rm = c_roundingModes[j].second;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                std::cerr << "ERROR: " << val_str << " is not a valid configuration for attribute " << name << "."
                          << std::endl;
                exit(1);
            }
        }

        else if (!stricmp(name, "stochasticRoundingMode"))
        {
            if (val == "0")
            {
                params->stochsaticConversion = false;
            }
            else if (val == "1")
            {
                params->stochsaticConversion = true;
            }
            else if (random)
            {
                params->stochsaticConversion = ((rand() & 0x1) == 1);
            }
        }

        else if (!stricmp(name, "skipRef"))
        {
            if (val == "0")
            {
                params->skipReference = false;
            }
            else if (val == "1")
            {
                params->skipReference = true;
            }
            else if (random)
            {
                params->skipReference = ((rand() & 0x1) == 1);
            }
            else
            {
                params->skipReference = false;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->skipReference = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->skipReference = true;
                        }
                        break;
                    }
                }
            }
        }

        else if (!stricmp(name, "scaledRandomValues"))
        {
            if (val == "0")
            {
                params->scaledRandomValues = false;
            }
            else if (val == "1")
            {
                params->scaledRandomValues = true;
            }
            else if (random)
            {
                params->scaledRandomValues = ((rand() & 0x1) == 1);
            }
            else
            {
                params->scaledRandomValues = false;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->scaledRandomValues = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->scaledRandomValues = true;
                        }
                        break;
                    }
                }
            }
        }

        else if (!stricmp(name, "fullDesc"))
        {
            params->fullDesc = random ? ((rand() & 0x1) == 1) : val == "1";
        }

        else if (!stricmp(name, "xInSram"))
        {
            if (val == "0")
            {
                params->xInSram = false;
            }
            else if (val == "1")
            {
                params->xInSram = true;
            }
            else if (random)
            {
                params->xInSram = ((rand() & 0x1) == 1);
            }
            else
            {
                params->xInSram = false;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->xInSram = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->xInSram = true;
                        }
                        break;
                    }
                }
            }
        }

        else if (!stricmp(name, "xInHbm"))
        {
            if (val == "0")
            {
                params->xInSram = true;
            }
            else if (val == "1")
            {
                params->xInSram = false;
            }
            else if (random)
            {
                params->xInSram = ((rand() & 0x1) == 1);
            }
            else
            {
                params->xInSram = true;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->xInSram = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->xInSram = false;
                        }
                        break;
                    }
                }
            }
        }

        else if (!stricmp(name, "yInSram"))
        {
            if (val == "0")
            {
                params->yInSram = false;
            }
            else if (val == "1")
            {
                params->yInSram = true;
            }
            else if (random)
            {
                params->yInSram = ((rand() & 0x1) == 1);
            }
            else
            {
                params->yInSram = false;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->yInSram = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->yInSram = true;
                        }
                    }
                }
            }
        }

        else if (!stricmp(name, "yInHbm"))
        {
            if (val == "0")
            {
                params->yInSram = true;
            }
            else if (val == "1")
            {
                params->yInSram = false;
            }
            else if (random)
            {
                params->yInSram = ((rand() & 0x1) == 1);
            }
            else
            {
                params->yInSram = true;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->yInSram = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->yInSram = false;
                        }
                        break;
                    }
                }
            }
        }

        else if (!stricmp(name, "wInSram"))
        {
            if (val == "0")
            {
                params->wInSram = false;
            }
            else if (val == "1")
            {
                params->wInSram = true;
            }
            else if (random)
            {
                params->wInSram = ((rand() & 0x1) == 1);
            }
            else
            {
                params->wInSram = false;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->wInSram = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->wInSram = true;
                        }
                    }
                }
            }
        }

        else if (!stricmp(name, "wInHbm"))
        {
            if (val == "0")
            {
                params->wInSram = true;
            }
            else if (val == "1")
            {
                params->wInSram = false;
            }
            else if (random)
            {
                params->wInSram = ((rand() & 0x1) == 1);
            }
            else
            {
                params->wInSram = true;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        if (val.find("random") != std::string::npos)
                        {
                            params->wInSram = ((rand() & 0x1) == 1);
                        }
                        else
                        {
                            params->wInSram = false;
                        }
                        break;
                    }
                }
            }
        }

        else if (!stricmp(name, "memsetOutput"))
        {
            if (val == "0")
            {
                params->memsetOutput = false;
            }
            else if (val == "1")
            {
                params->memsetOutput = true;
            }
            else if (random)
            {
                params->memsetOutput = ((rand() & 0x1) == 1);
            }
            else
            {
                params->memsetOutput = false;
                for (int op = 0; op < sizeof(c_op) / sizeof(c_op[0]); op++)
                {
                    if ((c_op[op].second == params->op) && (val.find(c_op[op].first) != std::string::npos))
                    {
                        params->memsetOutput = true;
                        break;
                    }
                }
            }
        }

        else if (!stricmp(name, "incDec"))
        {
            params->incDec = (val == "1");
        }

        else if (!stricmp(name, "loop"))
        {
            params->loop = (val == "1");
        }

        else if (!stricmp(name, "sramStreamingBudget"))
        {
            params->sramStreamingBudget = parseInt(val);
            MME_ASSERT(params->sramStreamingBudget == -1, "streaming budget should not be defined");
        }

        else if (!stricmp(name, "xMinVal"))
        {
            float tmp = atof(val_str);
            params->xMinVal = params->fp ? *(int32_t*) &tmp : parseInt(val);
        }

        else if (!stricmp(name, "xMaxVal"))
        {
            float tmp = atof(val_str);
            params->xMaxVal = params->fp ? *(int32_t*) &tmp : parseInt(val);
        }

        else if (!stricmp(name, "yMinVal"))
        {
            float tmp = atof(val_str);
            params->yMinVal = params->fp ? *(int32_t*) &tmp : parseInt(val);
        }

        else if (!stricmp(name, "yMaxVal"))
        {
            float tmp = atof(val_str);
            params->yMaxVal = params->fp ? *(int32_t*) &tmp : parseInt(val);
        }

        else if (!stricmp(name, "wMinVal"))
        {
            float tmp = atof(val_str);
            params->wMinVal = params->fp ? *(int32_t*) &tmp : parseInt(val);
        }

        else if (!stricmp(name, "wMaxVal"))
        {
            float tmp = atof(val_str);
            params->wMaxVal = params->fp ? *(int32_t*) &tmp : parseInt(val);
        }

        else if (!stricmp(name, "id"))
        {
            params->tid = parseInt(val);
            params->setId = false;
        }

        else
        {
            std::cerr << "ERROR: unknown configuration " << name << ".\n";
            exit(1);
        }
    }

    fixParamConflicts(params);

    return true;
}

static void cfgProduct(const std::vector<Cfg>* globalCfgs,
                       const std::vector<std::vector<Cfg>>* testsCfgs,
                       std::vector<MmeTestParams_t>* testsParams)
{
    MmeTestParams_t gp = {0};
    parseGlobalParams(globalCfgs, &gp);

    for (auto testCfg : *testsCfgs)
    {
        std::vector<int> counters(testCfg.size(), 0);
        int idx = counters.size();
        while (idx >= 0)
        {
            if (idx == counters.size())
            {
                MmeTestParams_t tp = gp;
                if (parseTestParams(&testCfg, &counters, &tp))
                {
                    testsParams->push_back(tp);
                }
                idx--;
            }
            else
            {
                const char* name = testCfg[idx].name.c_str();
                counters[idx]++;
                if ((counters[idx] == testCfg[idx].val.size()) || !stricmp(name, "xSizes") ||
                    !stricmp(name, "ySizes") || !stricmp(name, "wSizes") || !stricmp(name, "dilation") ||
                    !stricmp(name, "strides") || !stricmp(name, "padding"))
                {
                    counters[idx] = 0;
                    idx--;
                }
                else
                {
                    idx = counters.size();
                }
            }
        }
    }
}

void cfgFile2Tests(const std::string& fileName, unsigned seed, std::vector<MmeTestParams_t>* testsParams)
{
    std::vector<Cfg> globalCfgs;
    std::vector<std::vector<Cfg>> testCfgs;

    getConfigLines(fileName, &globalCfgs, &testCfgs);
    cfgProduct(&globalCfgs, &testCfgs, testsParams);

    if ((*testsParams)[0].shuffle)
    {
        for (int i = 0; i < testsParams->size(); i++)
        {
            int j = rand() % testsParams->size();
            std::swap((*testsParams)[i], (*testsParams)[j]);
        }
    }

    for (int i = 0; i < testsParams->size(); i++)
    {
        if ((*testsParams)[i].setId)
        {
            (*testsParams)[i].tid = i;
        }
        (*testsParams)[i].seed = seed;
    }
}

static std::string intArr2str(const int* arr, int size)
{
    std::stringstream sstream;
    for (int i = 0; i < size; i++)
    {
        sstream << arr[i];
        if (i != (size - 1))
        {
            sstream << ", ";
        }
    }
    return sstream.str();
}

std::string test2text(const MmeTestParams_t* tp)
{
    std::list<std::string> rows;

    rows.push_back(std::string("global:pole=") + ((tp->pole == NORTH_POLE) ? "north" : "south"));

    std::stringstream sramBaseStream;
    sramBaseStream << std::hex << tp->sramBase;
    rows.push_back(std::string("global:sramBase=0x") + sramBaseStream.str());

    std::stringstream sramSizeStream;
    sramSizeStream << std::hex << tp->sramSize;
    rows.push_back(std::string("global:sramSize=0x") + sramSizeStream.str());

    std::stringstream hbmBaseStream;
    hbmBaseStream << std::hex << tp->hbmBase;
    rows.push_back(std::string("global:hbmBase=0x") + hbmBaseStream.str());

    std::stringstream smBaseStream;
    smBaseStream << std::hex << tp->smBase;
    rows.push_back(std::string("global:smBase=0x") + smBaseStream.str());

    rows.push_back(std::string("global:multiplTests=0"));
    rows.push_back(std::string("global:fp=") + (tp->fp ? "1" : "0"));
    rows.push_back(std::string("global:shuffle=0"));
    rows.push_back(std::string("global:programInSram=") + (tp->programInSram ? "1" : "0"));

    rows.push_back(std::string(""));
    rows.push_back(std::string("testName=") + tp->name);

    for (int i = 0; i < (sizeof(c_op) / sizeof(c_op[0])); i++)
    {
        if (tp->op == c_op[i].second)
        {
            rows.push_back(std::string("operation=") + c_op[i].first);
            break;
        }
    }

    rows.push_back(std::string("reluEn=") + (tp->conv.relu ? "1" : "0"));
    rows.push_back(std::string("lowerEn=") + (tp->lower ? "1" : "0"));
    rows.push_back(std::string("sbReuse=") + (tp->sbReuse ? "1" : "0"));
    rows.push_back(std::string("unrollEn=") + (tp->unrollEn ? "1" : "0"));
    rows.push_back(std::string("dedwAsBgemmEn=") + (tp->dedwAsBgemmEn ? "1" : "0"));
    rows.push_back(std::string("recurringMisalignmentOptEn=") + (tp->recurringMisalignmentOptEn ? "1" : "0"));
    rows.push_back(std::string("adjustFloatShapes=") + (tp->adjustFloatShapes ? "1" : "0"));
    rows.push_back(std::string("sbReuseInStripes=") + (tp->sbReuseInStripes ? "1" : "0"));
    rows.push_back(std::string("signalPartial=") + (tp->signalPartial ? "1" : "0"));
    rows.push_back(std::string("memsetVoidPixels=") + (tp->memsetVoidPixels ? "1" : "0"));

    std::stringstream repeatsStream;
    repeatsStream << tp->repeats;
    rows.push_back(std::string("repeats=") + repeatsStream.str());

    rows.push_back(std::string("xSizes=") + intArr2str(tp->inputShape, tp->ioDim));
    rows.push_back(std::string("ySizes=") + intArr2str(tp->outputShape, tp->ioDim));
    rows.push_back(std::string("wSizes=") + intArr2str(tp->weightsShape, tp->conv.dim + 2));
    rows.push_back(std::string("dilation=") + intArr2str(tp->conv.dilation.data(), tp->conv.dim));
    rows.push_back(std::string("strides=") + intArr2str(tp->conv.convStride.data(), tp->conv.dim));
    rows.push_back(std::string("padding=") + intArr2str(tp->conv.padding.data(), tp->conv.dim));
    rows.push_back(std::string("inType=") + (tp->inputType == e_type_bf16 ? "bfloat" : "float"));
    rows.push_back(std::string("outType=") + (tp->outputType == e_type_bf16 ? "bfloat" : "float"));
    if (tp->fp)
    {
        if (tp->inputType == e_type_fp32)
        {
            rows.push_back(std::string("paddingValue=") + std::to_string(*(float*) &tp->conv.paddingValue));
        }
        else
        {
            int32_t tmp = tp->conv.paddingValue.f32 << 16;
            rows.push_back(std::string("paddingValue=") + std::to_string(*(float*) &tmp));
        }
    }
    else
    {
        if (tp->inputType == e_type_fp32)
        {
            rows.push_back(std::string("paddingValue=") + std::to_string(tp->conv.paddingValue.f32));
        }
        else
        {
            rows.push_back(std::string("paddingValue=") + std::to_string(tp->conv.paddingValue.bf16));
        }
    }

    if (tp->op == E_CONV_TEST_FWD || tp->op == E_CONV_TEST_DEDX)
    {
        for (int i = 0; i < (sizeof(c_convStrategies) / sizeof(c_convStrategies[0])); i++)
        {
            if (tp->pattern == c_convStrategies[i].second)
            {
                rows.push_back(std::string("convPattern=") + c_convStrategies[i].first);
                break;
            }
        }
    }
    else
    {
        rows.push_back(std::string("convPattern=ksf"));
    }

    if ((tp->op == E_CONV_TEST_AB) || (tp->op == E_CONV_TEST_ABT) || (tp->op == E_CONV_TEST_ATB) ||
        (tp->op == E_CONV_TEST_ATBT))
    {
        for (int i = 0; i < (sizeof(c_bgemmStrategies) / sizeof(c_bgemmStrategies[0])); i++)
        {
            if (tp->pattern == c_bgemmStrategies[i].second)
            {
                rows.push_back(std::string("bgemmPattern=") + c_bgemmStrategies[i].first);
                break;
            }
        }
    }
    else
    {
        rows.push_back(std::string("bgemmPattern=ckf"));
    }

    if (tp->op == E_CONV_TEST_DEDW)
    {
        for (int i = 0; i < (sizeof(c_dedwStrategies) / sizeof(c_dedwStrategies[0])); i++)
        {
            if (tp->pattern == c_dedwStrategies[i].second)
            {
                rows.push_back(std::string("dedwPattern=") + c_dedwStrategies[i].first);
                break;
            }
        }
    }
    else
    {
        rows.push_back(std::string("dedwPattern=kfc"));
    }

    for (int i = 0; i < (sizeof(c_geometry) / sizeof(c_geometry[0])); i++)
    {
        if (tp->geometry == c_geometry[i].second)
        {
            rows.push_back(std::string("geometry=") + c_geometry[i].first);
            break;
        }
    }

    for (int i = 0; i < (sizeof(c_roundingModes) / sizeof(c_roundingModes[0])); i++)
    {
        if (tp->rm == c_roundingModes[i].second)
        {
            rows.push_back(std::string("roundingMode=") + c_roundingModes[i].first);
            break;
        }
    }
    rows.push_back(std::string("stochasticRoundingMode=") + (tp->stochsaticConversion ? "1" : "0"));

    rows.push_back(std::string("xInSram=") + (tp->xInSram ? "1" : "0"));
    rows.push_back(std::string("yInSram=") + (tp->yInSram ? "1" : "0"));
    rows.push_back(std::string("wInSram=") + (tp->wInSram ? "1" : "0"));
    rows.push_back(std::string("memsetOutput=") + (tp->memsetOutput ? "1" : "0"));
    rows.push_back(std::string("incDec=") + (tp->incDec ? "1" : "0"));
    rows.push_back(std::string("loop=") + (tp->loop ? "1" : "0"));
    rows.push_back(std::string("fullDesc=") + (tp->fullDesc ? "1" : "0"));
    rows.push_back(std::string("skipRef=") + (tp->skipReference ? "1" : "0"));
    rows.push_back(std::string("scaledRandomValues=") + (tp->scaledRandomValues ? "1" : "0"));
    rows.push_back(std::string("sramStreamingBudget=") + std::to_string(tp->sramStreamingBudget));

    rows.push_back(std::string("xMinVal=") +
                   (tp->fp ? std::to_string(*(float*) &tp->xMinVal) : std::to_string(tp->xMinVal)));
    rows.push_back(std::string("xMaxVal=") +
                   (tp->fp ? std::to_string(*(float*) &tp->xMaxVal) : std::to_string(tp->xMaxVal)));
    rows.push_back(std::string("yMinVal=") +
                   (tp->fp ? std::to_string(*(float*) &tp->yMinVal) : std::to_string(tp->yMinVal)));
    rows.push_back(std::string("yMaxVal=") +
                   (tp->fp ? std::to_string(*(float*) &tp->yMaxVal) : std::to_string(tp->yMaxVal)));
    rows.push_back(std::string("wMinVal=") +
                   (tp->fp ? std::to_string(*(float*) &tp->wMinVal) : std::to_string(tp->wMinVal)));
    rows.push_back(std::string("wMaxVal=") +
                   (tp->fp ? std::to_string(*(float*) &tp->wMaxVal) : std::to_string(tp->wMaxVal)));
    rows.push_back(std::string("id=") + std::to_string(tp->tid));
    rows.push_back(std::string("//seed=") + std::to_string(tp->seed));

    std::string ret = "";
    for (auto& row : rows)
    {
        ret += row + "\n";
    }

    return ret;
}
