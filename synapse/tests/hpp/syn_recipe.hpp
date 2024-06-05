#pragma once

#include "syn_object.hpp"

namespace syn
{
class Recipe : public SynObject<synRecipeHandle>
{
public:
    Recipe() = default;

    Recipe(const std::string& recipeFilePath) : SynObject(createHandle<synRecipeHandle>(synRecipeDestroy))
    {
        SYN_CHECK(synRecipeDeSerialize(handlePtr(), recipeFilePath.c_str()));
    }

    void serialize(const std::string& recipeFilePath) const
    {
        SYN_CHECK(synRecipeSerialize(handle(), recipeFilePath.c_str()));
    }

    std::vector<uint64_t> getAttributes(const std::vector<synRecipeAttribute>& attributes) const
    {
        std::vector<uint64_t> ret(attributes.size());
        SYN_CHECK(synRecipeGetAttribute(ret.data(), attributes.data(), attributes.size(), handle()));
        return ret;
    }

    uint64_t getWorkspaceSize() const
    {
        uint64_t workspaceSize = 0;
        SYN_CHECK(synWorkspaceGetSize(&workspaceSize, handle()));
        return workspaceSize;
    }

    uint32_t getLaunchTensorsAmount() const
    {
        uint32_t numOfTensors;
        SYN_CHECK(synTensorRetrieveLaunchAmount(handle(), &numOfTensors));
        return numOfTensors;
    }

    std::vector<uint64_t> getLaunchTensorsIds() const
    {
        uint32_t              numOfTensors = getLaunchTensorsAmount();
        std::vector<uint64_t> ids(numOfTensors);
        SYN_CHECK(synTensorRetrieveLaunchIds(handle(), ids.data(), numOfTensors));
        return ids;
    }

    std::vector<synRetrievedLaunchTensorInfo> getLaunchTensorsInfo() const
    {
        const std::vector<uint64_t> ids = getLaunchTensorsIds();

        std::vector<synRetrievedLaunchTensorInfo> tensorsInfo;
        tensorsInfo.reserve(ids.size());
        for (const auto& id : ids)
        {
            tensorsInfo.emplace_back();
            tensorsInfo.back().tensorId = id;
        }

        SYN_CHECK(synTensorRetrieveLaunchInfoById(handle(), ids.size(), tensorsInfo.data()));

        return tensorsInfo;
    }

    std::vector<synRetrievedLaunchTensorInfoExt> getLaunchTensorsInfoExt() const
    {
        const std::vector<uint64_t> ids = getLaunchTensorsIds();

        std::vector<synRetrievedLaunchTensorInfoExt> tensorsInfo;
        tensorsInfo.reserve(ids.size());
        for (const auto& id : ids)
        {
            tensorsInfo.emplace_back();
            tensorsInfo.back().tensorId = id;
        }

        SYN_CHECK(synTensorRetrieveLaunchInfoByIdExt(handle(), ids.size(), tensorsInfo.data()));

        return tensorsInfo;
    }

    uint64_t sectionGetProp(const synSectionId sectionId, const synSectionProp prop) const
    {
        uint64_t propertyValue;
        SYN_CHECK(synRecipeSectionGetProp(handle(), sectionId, prop, &propertyValue));
        return propertyValue;
    }

private:
    Recipe(const std::shared_ptr<synRecipeHandle>& handle) : SynObject(handle) {}

    friend class GraphBase;  // GraphBase class requires access to Recipe private constructor
};
}  // namespace syn
