#include "data_types/non_standard_dtypes.h"
#include "config_parser_tests.h"
#include "mme_verification/common/config_parser.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

void MmeUTConfigParser::compareJsons(const json& testedJson, const json& referenceJson)
{
    // testedJson has default values which are not present in reference json
    ASSERT_GE(testedJson.size(), referenceJson.size());
    // iterate over the reference json (original), and check that all items exists in the re-parsed one.
    for (auto& item : referenceJson.items())
    {
        std::string refKey = item.key();
        const json& refValue = item.value();
        ASSERT_EQ(testedJson.count(refKey), 1);
        if (refKey == "tests")
        {
            compareTests(testedJson["tests"], referenceJson["tests"]);
        }
        else if (canSkip(refKey)) continue;
        else if (refValue.is_string())
        {
            compareString(testedJson[refKey], referenceJson[refKey], refKey);
        }
        else if (refValue.is_array())
        {
            compareArray(testedJson[refKey], referenceJson[refKey], refKey);
        }
        else
        {
            ASSERT_TRUE(refValue.is_object()) << "key : " << refKey << " file : " << m_currentFile;
            const json& nestedTestObj = testedJson[refKey];
            compareJsons(nestedTestObj, refValue);
        }
    }
}

void MmeUTConfigParser::compareTests(const json& testedJson, const json& referenceJson)
{
    ASSERT_TRUE(testedJson.is_array() && referenceJson.is_array()) << "file : " << m_currentFile;
    // currently this will compare only the first test in case of multiple values.
    const std::vector<json>& referenceVec = referenceJson.begin().value();
    const std::vector<json>& testedVec = testedJson.begin().value();
    for (const auto& attr : referenceVec)
    {
        const auto& refKey = attr.items().begin().key();
        const auto& refValue = attr.items().begin().value();
        if (canSkip(refKey))
        {
            continue;
        }
        const json& nestedRef = attr;
        auto it = std::find_if(testedVec.begin(), testedVec.end(), [&](const json& j) {
            return j.items().begin().key() == refKey;
        });
        ASSERT_NE(it, testedVec.end()) << "key : " << refKey << " file : " << m_currentFile;
        const json& nestedTested = *it;
        compareJsons(nestedTested, nestedRef);
    }
}

void MmeUTConfigParser::compareString(const json& testedJson, const json& referenceJson, std::string& key)
{
    ASSERT_TRUE(testedJson.is_string() && referenceJson.is_string()) << "key : " << key << " file : " << m_currentFile;
    const std::string& testedStr = testedJson.get<std::string>();
    const std::string& refStr = referenceJson.get<std::string>();
    if (canSkip(testedStr) || canSkip(refStr))
    {
        return;
    }
    ASSERT_EQ(testedStr, refStr) << "key : " << key << " file : " << m_currentFile;
}

void MmeUTConfigParser::compareArray(const json& testedJson, const json& referenceJson, std::string& key)
{
    ASSERT_TRUE(testedJson.is_array() && referenceJson.is_array()) << "key : " << key << " file : " << m_currentFile;
    const auto& referenceStringVec = referenceJson.get<std::vector<std::string>>();
    const auto& testedStringVec = testedJson.get<std::vector<std::string>>();
    // reference can contain multiple values.
    if (key.substr(1, 5) != "Sizes")
    {
        ASSERT_GE(referenceStringVec.size(), testedStringVec.size()) << "key : " << key << " file : " << m_currentFile;
        ;
    }
    unsigned minVecSize = std::min(testedStringVec.size(), referenceStringVec.size());
    for (unsigned i = 0; i < minVecSize; i++)
    {
        if (canSkip(referenceStringVec[i]))
        {
            continue;
        }
        if (referenceStringVec[i] == "fp32_non_ieee" && testedStringVec[i] == "fp32")
        {
            continue;
        }
        if (referenceStringVec[i] == "asInput")
        {
            continue;
        }
        ASSERT_EQ(testedStringVec[i], referenceStringVec[i]) << "key : " << key << " file : " << m_currentFile;
    }
}

bool MmeUTConfigParser::canSkip(const std::string& attr)
{
    return (attr == "random" || attr == "random_conv" || attr == "random_bgemm" || attr == "comment" || attr == "id");
}

TEST_F(MmeUTConfigParser, test_all_json_validity)
{
    char* mme_root = getenv("MME_ROOT");
    std::string configPath = mme_root;
    configPath += "/mme_verification/";
    for (std::string dirPath : {configPath+"common", configPath+"gaudi2", configPath+"gaudi3"})
    {
        testValidity(dirPath);
    }
}

void MmeUTConfigParser::testValidity(std::string& dirPath)
{
    unsigned jsonsValidated = 0;
    if (!checkDirExists(dirPath))
    {
        return;
    }

    std::cout << "Validating jsons files from directory: " << dirPath << std::endl;
    for (const fs::directory_entry& file : fs::recursive_directory_iterator(dirPath))
    {
        MmeCommon::MMEConfigPrinter printer(true /*dumpDefaults*/);
        MmeCommon::MMETestConfigParser parser(MmeCommon::e_mme_Gaudi2);
        fs::path filePath = file.path();
        if (filePath.extension() != ".json" || filePath.stem() == "config_all") continue;
        m_currentFile = filePath.string();
        std::ifstream configFile(filePath.string());
        ASSERT_TRUE(configFile.good()) << "file " << m_currentFile << " could not be opened";

        // parse reference from original config
        json reference = json::parse(configFile);
        // parse config, and print it back to a string
        parser.setSeed(1);
        ASSERT_TRUE(parser.parseJsonFile(filePath.string(), 1));

        std::string parsedJsonStr = printer.dumpAndSerialize(parser.getParsedTests());
        // re-parse to json. compare each key
        json reparsedTest = json::parse(parsedJsonStr);
        compareJsons(reparsedTest, reference);
    }
}

bool MmeUTConfigParser::checkDirExists(const std::string& dirPath)
{
    if (!fs::exists(fs::path(dirPath)))
    {
        std::cerr << "Directory: " << dirPath << " not found. skipping..." << std::endl;
        return false;
    }
    return true;
}
