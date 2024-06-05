#include "config_parser.h"
#include "mme_unit_test.h"
#include "json.hpp"
#include <istream>

using json = nlohmann::json;

class MmeUTConfigParser : public MMEUnitTest
{
public:
    void testValidity(std::string& dirPath);

protected:
    bool checkDirExists(const std::string& dirPath);
    void compareJsons(const json& lhs, const json& rhs);
    void compareTests(const json& testedJson, const json& referenceJson);
    void compareString(const json& testedJson, const json& referenceJson, std::string& key);
    void compareArray(const json& testedJson, const json& referenceJson, std::string& key);
    bool canSkip(const std::string& attr);
    std::string m_currentFile = {};
};
