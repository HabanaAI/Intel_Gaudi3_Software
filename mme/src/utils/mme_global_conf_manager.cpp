
#include <string>

#include "include/utils/mme_global_conf_manager.h"

MmeGlobalConfManager MmeGlobalConfManager::instance()
{
    return {};
}

#ifdef SWTOOLS_DEP
#include <hl_gcfg/hlgcfg.hpp>
#include <hl_gcfg/hlgcfg_item.hpp>
#include "mme_common/mme_global_conf.h"
#include "logger.h"

void MmeGlobalConfManager::init(const std::string& fileName)
{
    hl_gcfg::reset();
    if (hl_gcfg::getEnableExperimentalFlagsValue())
    {
        if (!fileName.empty())
        {
            auto fileExist = hl_gcfg::loadFromFile(fileName);
            if (!fileExist)
            {
                HLGCFG_LOG_INFO("Not using Global Configurations file ({}). {}", fileName, fileExist);
            }
        }
    }

    // Create "used" file:
    if (GCFG_MME_CREATE_USED_CONFIGS_FILE.value())
    {
        std::string usedFileName = hl_logger::getLogsFolderPath() + "/mme.used";
        hl_gcfg::saveToFile(usedFileName);
    }
}

#else
void MmeGlobalConfManager::init(const std::string& fileName) {}
#endif