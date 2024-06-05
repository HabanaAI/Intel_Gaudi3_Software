#include "json_utils.h"

#include "spdlog/fmt/bundled/core.h"
#include "types_exception.h"
#include "utils.h"
#include "filesystem.h"

#include <fstream>

namespace json_utils
{
Json jsonFromFile(const std::string& filePath)
{
    if (!fs::exists(filePath) || fs::file_size(filePath) == 0) return Json();

    validateFilePath(filePath);

    Json          ret;
    std::ifstream jsonFile(filePath);
    if (jsonFile.good())
    {
        jsonFile >> ret;
    }
    return ret;
}

void jsonToFile(const Json& jsonObject, const std::string& filePath, unsigned indent)
{
    validateFilePath(filePath);

    std::ofstream jsonFile;
    jsonFile.open(filePath, std::ios::trunc);
    if (jsonFile.fail())
    {
        throw SynapseException(fmt::format("failed to write file: {}", filePath));
    }
    jsonFile << jsonObject.dump(indent);
}

const Json& get(const Json& j, const std::string& f)
{
    if (j.count(f) != 0)
    {
        return j.at(f);
    }
    throw SynapseException(fmt::format("missing required field: {}", f));
}

std::string toString(const json_utils::Json& j)
{
    std::ostringstream ss;
    ss << j;
    return ss.str();
}
}  // namespace json_utils
