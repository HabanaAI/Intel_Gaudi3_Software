#include "utils.h"
#include "hl_logger/hllog_core.hpp"
#include <fstream>
/*
 * test crash handlers
 * can be tested only manually and one test at a time
 * each test crashes and in the log file it should add the log message from the test
 * */
int recursion(int i)
{
    int* v = (int*)alloca(100 * 1024);
    v[0]   = i * i;
    v[1]   = i * i + 1;
    if (i > 1)
    {
        return recursion(i - 1) + v[i & 1];
    }
    return 1;
}

std::vector<std::string> findStringsInLog(const std::string& strToFind, const std::string& filename, const std::string& logsDir)
{
    const int searchRegion = 10000;  // search within [EndOfFile - searchRegion, EndOfFile]

    std::ifstream ifs(logsDir + "/" + filename);

    ifs.seekg(0, std::ios_base::end);
    if (ifs.tellg() > searchRegion)
    {
        ifs.seekg(-searchRegion, std::ios_base::end);
    }
    else
    {
        ifs.seekg(0, std::ios_base::beg);
    }
    std::string str;
    std::vector<std::string> matchLines;
    while (ifs.good() && !ifs.eof())
    {
        std::getline(ifs, str);
        if (str.find(strToFind) != std::string::npos)
        {
            matchLines.push_back(str);
        }
    }
    return matchLines;
}

std::vector<std::string> findStringsInLog(const std::string& strToFind, const std::string& filename)
{
    std::string logsFolder = hl_logger::getLogsFolderPath();
    return findStringsInLog(strToFind, filename, logsFolder);
}

int findStringsCountInLog(const std::string& strToFind, const std::string& filename)
{
    return findStringsInLog(strToFind, filename).size();
}

std::string filenameFromPath(const std::string& path)
{
    auto pos = path.rfind("/");
    if (pos == std::string::npos)
    {
        return "";
    }

    return path.substr(pos);
}
