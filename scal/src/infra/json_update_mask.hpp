#pragma once

#include "common/json.hpp"

class JsonUpdateMask
{
public:
    struct EngMasks
    {
        static uint64_t constexpr MAX64 = std::numeric_limits<uint64_t>::max();

        uint64_t tpc  = MAX64;
        uint64_t mme  = MAX64;
        uint64_t arc  = MAX64;
        uint64_t nic  = MAX64;
        uint64_t pdma = MAX64;
        uint64_t rot  = MAX64;
        uint64_t dma  = MAX64;
    };

    JsonUpdateMask(const scaljson::json& json, int fd);

    void run();
    bool initMasks();

    EngMasks&      testingOnlyGetEngMasks() { return m_engMasks; };
    scaljson::json getPatch();
    scaljson::json getItemPatch() { return m_overrideItems; };
    scaljson::json getSectionPatch() { return m_overrideSections; };

private:
    bool        getEngMasks();
    bool        iterateJson(const scaljson::json& j, const std::string& path);

    bool        isTpcNotInMask(std::string& s);
    bool        isMmeNotInMask(std::string& s);
    bool        isNicNotInMask(std::string& s);
    bool        isRotNotInMask(std::string& s);
    bool        isDmaNotInMask(std::string& s);

    std::string getFirstValidEng();

    bool        handleTpcPolicy(const std::string& item, const std::string& path);
    bool        removeMme(const std::string& item, const std::string& path);
    bool        removeNic(const std::string& item, const std::string& path);
    bool        removeRot(const std::string& item, const std::string& path);
    bool        removeDma(const std::string& item, const std::string& path);

    void        addToItemPatchJson(const scaljson::json& patch_operation, const std::string& path, const std::string& item);
    void        addToSectionPatchJson(const scaljson::json& patch_operation, const std::string& path, const std::string& item);

    const scaljson::json& m_json;
    scaljson::json        m_overrideItems    = scaljson::json::array();
    scaljson::json        m_overrideSections = scaljson::json::array();
    EngMasks              m_engMasks;
    int                   m_fd;
};

