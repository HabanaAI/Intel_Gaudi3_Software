#include <iostream>
#include <hlthunk.h>
#include "json_update_mask.hpp"
#include "logger.h"

/*
 ***************************************************************************************************
 *   @brief pathUp() - Gets a stirng with the path and returns a string of one path up (until the first "/" from the end
 ***************************************************************************************************
*/
static std::string pathUp(const std::string& path)
{
    std::size_t pos = path.find_last_of("/");
    if (pos == std::string::npos)
    {
        assert(0);
        LOG_ERR(SCAL, "Auto json: can't go up from path {}", path);
    }

    std::string rtn = path.substr(0, pos);
    return rtn;
}

/*
 ***************************************************************************************************
 *   @brief endsWith() - checks if s string "str" ends with string "suffix"
 ***************************************************************************************************
*/
static bool endsWith(const std::string& str, const std::string& suffix)
{
    if (str.length() < suffix.length())
{
        return false;
    }

    return str.substr(str.length() - suffix.length()) == suffix;
}

/*
 ***************************************************************************************************
 *   @brief getTpcNum() - gets a string in the format of TPC_x_y and return the TPC number (0-63)
 ***************************************************************************************************
*/
static int getTpcNum(const std::string& s)
{
    if (s.find("TPC") == std::string::npos)
    {
        return -1;
    }

    size_t pos1 = s.find('_');
    if (pos1 == std::string::npos)
    {
        return -1;
    }
    size_t pos2 = s.find('_', pos1 + 1);
    if (pos2 == std::string::npos)
    {
        return -1;
    }

    // Extract num1 and num2 substrings
    std::string num1_str = s.substr(pos1 + 1, pos2 - pos1 - 1);
    std::string num2_str = s.substr(pos2 + 1);

    // Convert num1 and num2 strings to integers
    int num1, num2;
    try {
        num1 = stoi(num1_str);
    }
    catch(...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi1 for TPC {} {}", s, num1_str);
        return -1;
    }
    try {
        num2 = stoi(num2_str);
    }
    catch (...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi2 for TPC {} {}", s, num2_str);
        return -1;
    }

    return num1 * 8 + num2;
}

/*
 ***************************************************************************************************
 *   @brief getMmeNum() - gets a string in the format of MME_x and return the MME number
 ***************************************************************************************************
*/
static int getMmeNum(const std::string& s)
{
    if (s.find("MME") == std::string::npos)
    {
        return -1;
    }

    size_t pos = s.find('_');
    if (pos == std::string::npos)
    {
        return -1;
    }
    std::string num_str = s.substr(pos + 1);

    // Convert num1 and num2 strings to integers
    int num;
    try {
        num = stoi(num_str);
    }
    catch(...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi for MME {} {}", s, num_str);
        return -1;
    }
    return num;
}

/*
 ***************************************************************************************************
 *   @brief getNicNum() - gets a string in the format of NIC_x_y and return the NIC number
 ***************************************************************************************************
*/
static int getNicNum(const std::string& s)
{
    if (s.find("NIC") == std::string::npos)
    {
        return -1;
    }

    size_t pos1 = s.find('_');
    if (pos1 == std::string::npos)
    {
        return -1;
    }
    size_t pos2 = s.find('_', pos1 + 1);
    if (pos2 == std::string::npos)
    {
        return -1;
    }

    // Extract num1 and num2 substrings
    std::string num1_str = s.substr(pos1 + 1, pos2 - pos1 - 1);
    std::string num2_str = s.substr(pos2 + 1);

    // Convert num1 and num2 strings to integers
    int num1, num2;
    try {
        num1 = stoi(num1_str);
    }
    catch(...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi1 for NIC {} {}", s, num1_str);
        return -1;
    }
    try {
        num2 = stoi(num2_str);
    }
    catch (...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi2 for NIC {} {}", s, num2_str);
        return -1;
    }

    return num1 * 6 + num2;
}

/*
 ***************************************************************************************************
 *   @brief getRotNum() - gets a string in the format of ROT_x_y and return the ROT number
 ***************************************************************************************************
*/
static int getRotNum(const std::string& s)
{
    if (s.find("ROT") == std::string::npos)
    {
        return -1;
    }

    size_t pos1 = s.find('_');
    if (pos1 == std::string::npos)
    {
        return -1;
    }
    size_t pos2 = s.find('_', pos1 + 1);
    if (pos2 == std::string::npos)
    {
        return -1;
    }

    // Extract num1 and num2 substrings
    std::string num1_str = s.substr(pos1 + 1, pos2 - pos1 - 1);
    std::string num2_str = s.substr(pos2 + 1);

    // Convert num1 and num2 strings to integers
    int num1, num2;
    try {
        num1 = stoi(num1_str);
    }
    catch(...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi1 for NIC {} {}", s, num1_str);
        return -1;
    }
    try {
        num2 = stoi(num2_str);
    }
    catch (...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi2 for NIC {} {}", s, num2_str);
        return -1;
    }

    return num1 / 2 * 2 + num2;
}

/*
 ***************************************************************************************************
 *   @brief getDmaNum() - gets a string in the format of EDMA_x_y and return the EDMA number
 ***************************************************************************************************
*/
static int getDmaNum(const std::string& s)
{
    if (s.find("EDMA") == std::string::npos)
    {
        return -1;
    }

    size_t pos1 = s.find('_');
    if (pos1 == std::string::npos)
    {
        return -1;
    }
    size_t pos2 = s.find('_', pos1 + 1);
    if (pos2 == std::string::npos)
    {
        return -1;
    }

    // Extract num1 and num2 substrings
    std::string num1_str = s.substr(pos1 + 1, pos2 - pos1 - 1);
    std::string num2_str = s.substr(pos2 + 1);

    // Convert num1 and num2 strings to integers
    int num1, num2;
    try {
        num1 = stoi(num1_str);
    }
    catch(...) {
        LOG_TRACE(SCAL, "failed to stoi1 for DMA {} {}", s, num1_str);
        return -1;
    }
    try {
        num2 = stoi(num2_str);
    }
    catch (...) {
        LOG_TRACE(SCAL, "Auto json: failed to stoi2 for DMA {} {}", s, num2_str);
        return -1;
    }

    return num1 / 2 * 2 + num2;
}

/*
 ***************************************************************************************************
 *   @brief JsonUpdateMask() - constructgor
 ***************************************************************************************************
*/
JsonUpdateMask::JsonUpdateMask(const scaljson::json& json, int fd) : m_json(json), m_fd(fd)
{

}

/*
 ***************************************************************************************************
 *   @brief getEngMasks() - get all the engine masks from hlthunk call
 ***************************************************************************************************
*/
bool JsonUpdateMask::getEngMasks()
{
    hlthunk_hw_ip_info hwIp {};

    int ret = hlthunk_get_hw_ip_info(m_fd, &hwIp);

    if (ret < 0)
    {
        LOG_ERR(SCAL, "Auto json: Failed reading hwIp with ret {} errno {}", ret, errno);
        assert(0);
        return false;
    }

    m_engMasks.tpc = hwIp.tpc_enabled_mask_ext;
    LOG_INFO(SCAL, "Auto json: tpc mask {:x}", m_engMasks.tpc);

    m_engMasks.mme = hwIp.mme_enabled_mask;
    LOG_INFO(SCAL, "Auto json: mme mask {:x}", m_engMasks.mme);

    m_engMasks.nic = hwIp.nic_ports_mask;
    LOG_INFO(SCAL, "Auto json: nic mask {:x}", m_engMasks.nic);

    m_engMasks.arc = hwIp.sched_arc_enabled_mask;
    LOG_INFO(SCAL, "Auto json: arc mask {:x}", m_engMasks.arc);

    m_engMasks.rot = hwIp.rotator_enabled_mask;
    LOG_INFO(SCAL, "Auto json: rot mask {:x}", m_engMasks.rot);

    LOG_INFO(SCAL, "Auto json: dma mask {:x}", m_engMasks.dma);

    m_engMasks.pdma = hwIp.pdma_user_owned_ch_mask;
    LOG_INFO(SCAL, "Auto json: pdma mask {:x}", m_engMasks.pdma);

    return true;
}

/*
 ***************************************************************************************************
 *   @brief initMasks() - get all the engine masks
 ***************************************************************************************************
*/
bool JsonUpdateMask::initMasks()
{
    bool ret = getEngMasks();
    if (!ret)
    {
        LOG_ERR(SCAL, "Auto json: Failed getting masks");
    }
    return ret;
}

/*
 ***************************************************************************************************
 *   @brief getPatch() - get the calculated patch file
 ***************************************************************************************************
*/
scaljson::json JsonUpdateMask::getPatch()
{
    auto rtn = m_overrideItems;
    rtn.insert(rtn.end(), m_overrideSections.begin(), m_overrideSections.end());

    return rtn;
}

/*
 ***************************************************************************************************
 *   @brief run() - calculate the patch file
 ***************************************************************************************************
*/
void JsonUpdateMask::run()
{
    LOG_INFO(SCAL, "Auto json: Update masks with tpc mask {:x}",  m_engMasks.tpc);
    LOG_INFO(SCAL, "Auto json: Update masks with mme mask {:x}",  m_engMasks.mme);
    LOG_INFO(SCAL, "Auto json: Update masks with nic mask {:x}",  m_engMasks.nic);
    LOG_INFO(SCAL, "Auto json: Update masks with arc mask {:x}",  m_engMasks.arc);
    LOG_INFO(SCAL, "Auto json: Update masks with rot mask {:x}",  m_engMasks.rot);
    LOG_INFO(SCAL, "Auto json: Update masks with dma mask {:x}",  m_engMasks.dma);
    LOG_INFO(SCAL, "Auto json: Update masks with pdma mask {:x}", m_engMasks.pdma);

    iterateJson(m_json, "");
}

/*
 ***************************************************************************************************
 *   @brief addToItemPatchJson() - adds a patch to the itemJson at the beginning of the json (so removals are from end to start)
 ***************************************************************************************************
*/
void JsonUpdateMask::addToItemPatchJson(const scaljson::json& patch_operation, const std::string& path, const std::string& item)
{
    LOG_TRACE(SCAL, "Auto json: {}) For item {} adding to item patch json\n{}", m_overrideItems.size(), item, path, patch_operation.dump(4));
    m_overrideItems.insert(m_overrideItems.begin(), patch_operation);
}

/*
 ***************************************************************************************************
 *   @brief addToItemPatchJson() - adds a patch to the sectionsJson at the beginning of the json (so removals are from end to start)
 ***************************************************************************************************
*/
void JsonUpdateMask::addToSectionPatchJson(const scaljson::json& patch_operation, const std::string& path, const std::string& item)
{
    LOG_TRACE(SCAL, "Auto json: {}) For item {} adding to section patch json\n{}", m_overrideItems.size(), item, path, patch_operation.dump(4));
    m_overrideSections.insert(m_overrideSections.begin(), patch_operation);
}

/*
 ***************************************************************************************************
 *   @brief handleTpcPolicy() - handle the case of TPC that is not in mask. If it is qman, replcace,
 *                              if not, remove
 ***************************************************************************************************
*/
bool JsonUpdateMask::handleTpcPolicy(const std::string& item, const std::string& path)
{
    if (endsWith(path, "qman"))
    {
        std::string firstTpc = getFirstValidEng();
        scaljson::json patch_operation = {
            { "op", "replace" },
            { "path", path },
            { "value", firstTpc }
        };
        addToItemPatchJson(patch_operation, path, item);
        return false;
    }
    else
    {
        scaljson::json patch_operation = {
            { "op", "remove" },
            { "path", path }
        };
        addToItemPatchJson(patch_operation, path, item);
        return true;
    }
}

/*
 ***************************************************************************************************
 *   @brief removeMme() - add a patch to remove mme
 ***************************************************************************************************
*/
bool JsonUpdateMask::removeMme(const std::string& item, const std::string& path)
{
    scaljson::json patch_operation = {
                { "op", "remove" },
                { "path", path }
        };
    addToItemPatchJson(patch_operation, path, item);
    return true;
}

/*
 ***************************************************************************************************
 *   @brief removeMme() - add a patch to remove nic (note, we remove one path up)
 ***************************************************************************************************
*/
bool JsonUpdateMask::removeNic(const std::string& item, const std::string& path)
{
    std::string pathToRemove = pathUp(path);

    scaljson::json patch_operation = {
                { "op", "remove" },
                { "path", pathToRemove }
        };

    addToItemPatchJson(patch_operation, pathToRemove, item);
    return true;
}

/*
 ***************************************************************************************************
 *   @brief removeMme() - add a patch to remove rot
 ***************************************************************************************************
*/
bool JsonUpdateMask::removeRot(const std::string& item, const std::string& path)
{
    scaljson::json patch_operation = {
                { "op", "remove" },
                { "path", path }
        };
    addToItemPatchJson(patch_operation, path, item);
    return true;
}

/*
 ***************************************************************************************************
 *   @brief removeMme() - add a patch to remove edma
 ***************************************************************************************************
*/
bool JsonUpdateMask::removeDma(const std::string& item, const std::string& path)
{
    scaljson::json patch_operation = {
        { "op", "remove" },
        { "path", path }
    };
    addToItemPatchJson(patch_operation, path, item);
    return true;
}

/*
 ***************************************************************************************************
 *   @brief iterateJson() - Go over the json recursively, replace/remove items as needed
 ***************************************************************************************************
*/
bool JsonUpdateMask::iterateJson(const scaljson::json& j, const std::string& path)
{
    if (j.is_object()) // iterate over all items in the object
    {
        bool anyRemoved = false;
        for (auto& element : j.items())
        {
            const std::string&    name     = element.key();
            const scaljson::json& subValue = element.value();
            std::string           subPath  = path + "/" + name;
            //            std::cout << "Path: " << subPath << std::endl;
            anyRemoved |= iterateJson(subValue, subPath);
        }
        return anyRemoved;
    }
    else if (j.is_array()) // iterate over all the items in the array
    {
        int removed = 0;
        for (int i = 0; i < j.size(); i++)
        {
            // if an item is removed, the items after it move "up". So the path is i-removed
            bool wasRemoved = iterateJson(j[i], path + "/" + std::to_string(i));
            if (wasRemoved)
            {
                removed++;
            }
        }
        if (j.size() == removed) // Array was fully removed, remove the whole path
        {
            std::string pathToRemove = pathUp(path);
            scaljson::json patch_operation = {
                { "op", "remove" },
                { "path", pathToRemove }
            };
            addToSectionPatchJson(patch_operation, path, "Array-is-empty");
        }
        return false;
    }
    else if (j.is_string()) // check if tpc/mme/nic/etc. and remove/replace if needed
    {
        std::string s = j.get<std::string>();
        if (isTpcNotInMask(s))
        {
            LOG_TRACE(SCAL, "Auto json: Found a TPC not in mask {}", s);
            return handleTpcPolicy(s, path);
        }

        if (isMmeNotInMask(s))
        {
            LOG_TRACE(SCAL, "Auto json: Found an MME not in mask {}", s);
            return removeMme(s, path);
        }

        if (isNicNotInMask(s))
        {
            LOG_TRACE(SCAL, "Auto json: Found an NIC not in mask {}", s);
            return removeNic(s, path);
        }

        if (isRotNotInMask(s))
        {
            LOG_TRACE(SCAL, "Auto json: Found an ROT not in mask {}", s);
            return removeRot(s, path);
        }

        if (isDmaNotInMask(s))
        {
            LOG_TRACE(SCAL, "Auto json: Found an DMA not in mask {}", s);
            return removeDma(s, path);
        }

        return false;
    }
    else if (j.is_number() || j.is_boolean() || j.is_null()) // nothing to do here
    {
        return false;
    }
    std::cout << "assert: path " << path << " " << j << std::endl;
    assert(0);
    return false;
}

/*
 ***************************************************************************************************
 *   @brief isTpcNotInMask() - Checks if TPC that is not in the mask
 ***************************************************************************************************
*/
bool JsonUpdateMask::isTpcNotInMask(std::string& s)
{
    int tpcNum = getTpcNum(s);
    if (tpcNum == -1) return false;

    bool rtn = (1ull << tpcNum) & m_engMasks.tpc;

    return !rtn;
}

/*
 ***************************************************************************************************
 *   @brief isMmeNotInMask() - Checks if MME that is not in the mask
 ***************************************************************************************************
*/
bool JsonUpdateMask::isMmeNotInMask(std::string& s)
{
    int mmeNum = getMmeNum(s);
    if (mmeNum == -1) return false;

    bool rtn = (1ull << mmeNum) & m_engMasks.mme;

    return !rtn;
}

/*
 ***************************************************************************************************
 *   @brief isNicNotInMask() - Checks if NIC that is not in the mask
 ***************************************************************************************************
*/
bool JsonUpdateMask::isNicNotInMask(std::string& s)
{
    int nicNum = getNicNum(s);
    if (nicNum == -1) return false;

    nicNum *= 2; // need to adjust to the mask
    bool rtn = (1ull << nicNum) & m_engMasks.nic;

    return !rtn;
}

/*
 ***************************************************************************************************
 *   @brief isRotNotInMask() - Checks if ROT that is not in the mask
 ***************************************************************************************************
*/
bool JsonUpdateMask::isRotNotInMask(std::string& s)
{
    int rotNum = getRotNum(s);
    if (rotNum == -1) return false;

    bool rtn = (1ull << rotNum) & m_engMasks.rot;

    return !rtn;
}

/*
 ***************************************************************************************************
 *   @brief isDmaNotInMask() - Checks if EDMA that is not in the mask
 ***************************************************************************************************
*/
bool JsonUpdateMask::isDmaNotInMask(std::string& s)
{
    int dmaNum = getDmaNum(s);
    if (dmaNum == -1) return false;

    bool rtn = (1ull << dmaNum) & m_engMasks.dma;

    return !rtn;
}

/*
 ***************************************************************************************************
 *   @brief getFirstValidEng() - In case we need to replace a TPC, select and exisiting engine to use
 ***************************************************************************************************
*/
std::string JsonUpdateMask::getFirstValidEng()
{
    for (int i = 0; i < 64; i++)
    {
        if ((1ull << i) & m_engMasks.tpc)
        {
            int hd = i / 8;
            int inHd = i % 8;

            return "TPC_" + std::to_string(hd) + "_" + std::to_string(inHd);
        }
    }

    if (m_engMasks.nic & 1)
    {
        return "NIC_0_0";
    }

    LOG_ERR(SCAL, "Auto json: no valid Eng");
    assert(0);
    return "????";
}
