#include "scal_basic_test.h"
#include "scal.h"
#include "hlthunk.h"
#include "logger.h"
#include "scal_test_utils.h"
#include <dirent.h>

typedef std::vector<std::string> strList;

int listdir(const char *path, strList& paths,const char* ext = nullptr)
{
    struct dirent *entry;
    DIR *dp;

    dp = opendir(path);
    if (dp == NULL)
    {
        LOG_ERR(SCAL," opendir({}) failed", path);
        return -1;
    }

    while ((entry = readdir(dp)))
    {
        if(ext)
        {
            char* foundExt = strrchr(entry->d_name,'.');
            if(!foundExt) continue;// no ext
            if(strcmp(ext,foundExt))
                continue;// wrong ext
        }
        std::string file = entry->d_name;
        paths.push_back(file);
    }
    closedir(dp);
    return 0;
}

int test_config_file(int scalFd, const char* configFilePath)
{
    scal_handle_t scalHandle;

    int rc = scal_init(scalFd, configFilePath, &scalHandle, nullptr);
    scal_destroy(scalHandle);
    return rc;
}

TEST_F(SCALTest, DISABLED_scheduler_config_test)
{
    int scalFd = hlthunk_open(HLTHUNK_DEVICE_DONT_CARE, NULL);
    if (scalFd < 0)
    {
        //todo report error " "Can't open a device file: errno = {}: {}", errno, std::strerror(errno));
        LOG_ERR(SCAL,"{}:Can't open a device file errno {} {}",__FUNCTION__,  errno, std::strerror(errno));
        assert(0);
        // todo error handling
    }
#ifdef NDEBUG
    // these tests feed wrong configuration files into scal
    // in debug, an assert would be thrown so we cannot catch it
    std::string configFolder = "tests/misconfigs/";
    strList paths;
    if(listdir(configFolder.c_str(), paths,".json"))
    {
        LOG_ERR(SCAL," listdir({}) failed", configFolder);
        assert(0);
    }
    std::sort(paths.begin(), paths.end());
    for (const auto & file : paths)
    {
        std::string fullPath = configFolder + file;
        printf("testing config %s\n",fullPath.c_str());
        int rc = test_config_file(scalFd, fullPath.c_str());
        if (rc == 0)
        {
            LOG_ERR(SCAL,"{}: scal init should have failed on {}" ,__FUNCTION__, file);
            assert(0);
        }
    }
#else
    printf("scheduler_config_test should run in Release mode\n");
#endif // NDEBUG

    hlthunk_close(scalFd);
}