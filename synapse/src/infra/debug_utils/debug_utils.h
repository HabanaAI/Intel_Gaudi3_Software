#pragma once

#include "fmt-9.1.0/include/fmt/core.h"
#include "infra/log_manager.h"
#include <fstream>

namespace debug_utils
{
void createMpmMarker()
{
    std::string logsFolder;
    synapse::LogManager::getLogsFolderPath(logsFolder);
    std::string   mpmMarkerFilePath = fmt::format("{}/mpm_debug_marker", logsFolder);
    std::ofstream mpmMarkerFile;
    mpmMarkerFile.open(mpmMarkerFilePath, std::fstream::app);
    mpmMarkerFile.close();
}
}  // namespace debug_utils

#define SET_MPM_MARKER() debug_utils::createMpmMarker()
