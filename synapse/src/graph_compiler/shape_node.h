#pragma once

#include "types.h"
#include "recipe_patch_point.h"
#include "recipe_program.h"
#include <unordered_map>
#include <algorithm>

class ShapeNode
{
public:
    using PostSifUpdates = std::vector<std::pair<TensorPtr, TensorPtr>>;

    ShapeNode(Node& node) : m_connectedNode(node) {}
    ShapeNode(const ShapeNode& other, Node& node) : m_connectedNode(node), m_patchPoints(other.m_patchPoints) {}

    ShapeNode& operator=(const ShapeNode& other);
    ShapeNode& operator=(ShapeNode&& other);

    ShapeNode(const ShapeNode& other) = delete;  // Must initialize parent object.
    ShapeNode(ShapeNode&& other)      = delete;  // Must initialize parent object.
    ~ShapeNode()                      = default;

    void addPatchPoint(DynamicPatchPointPtr patchPoint);

    // After the SIF is invoked copy the shape from src to dst.
    // This is intended to update non-inferable siblings like internally created broadcast and memset nodes.
    void addPostSifUpdate(const TensorPtr& src, const TensorPtr& dst);

    const PostSifUpdates& getPostSifUpdates() const { return m_postSifUpdate; }

    void serialize(const ShapePlaneInfoContainer& shapePlaneInfoContainer,
                   shape_plane_node_t&            serializeNode,
                   RecipeAllocator*               pRecipeAlloc);

    void serializeRoi(const NodeROI& roi, roi_info_t& serializeRoi, RecipeAllocator* pRecipeAlloc);
    void serializeInOutRoi(const TensorROIVector& tensorRois,
                           uint32_t&              size,
                           tensor_roi_t*&         convertedRois,
                           RecipeAllocator*       pRecipeAlloc);

    template<typename F>
    void eraseDynamicPatchPointsByPredicate(F&& f)
    {
        m_patchPoints.erase(std::remove_if(m_patchPoints.begin(), m_patchPoints.end(), f), m_patchPoints.end());
    }

    const std::vector<DynamicPatchPointPtr>& getPatchPoints() { return m_patchPoints; }

private:

    Node& m_connectedNode;
    std::vector<DynamicPatchPointPtr> m_patchPoints;
    PostSifUpdates m_postSifUpdate;

    struct AddPatchPointCache
    {
        bool                               initialized = false;
        std::list<NodeROI>::const_iterator m_lastIterator;
        size_t                             m_lastIndex;
    } m_addPatchPointCache;
};

using ShapeNodePtr = std::shared_ptr<ShapeNode>;
