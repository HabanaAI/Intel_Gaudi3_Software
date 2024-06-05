#pragma once

#include <string>

class MmeGlobalConfManager
{
public:
    static MmeGlobalConfManager instance();
    /**
     * Load global configuration from file
     * If file doesn't exist, create one with default values
     */
    static void init(const std::string& fileName);
};