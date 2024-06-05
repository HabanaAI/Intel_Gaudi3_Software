#pragma once

#include <recipe.h>
#include "synapse_types.h"
#include "params_file_manager.h"
#include "synapse_runtime_logging.h"

struct recipe_t;
struct shape_plane_graph_t;
class RecipeAllocator;

class RecipeSerializer
{
public:
    virtual ~RecipeSerializer() = default;

    static synStatus serialize(const recipe_t* pRecipe, const shape_plane_graph_t* spr, ParamsManagerBase* pParams);

    static synStatus
    deserialize(recipe_t* pRecipe, shape_plane_graph_t*& spr, ParamsManager* pParams, RecipeAllocator* pRecipeAlloc);

private:
    static bool verifyDsdVersion(shape_plane_graph_t* spr);
};

template<class RecipeOp>
class ScanRecipe
{
public:
    ScanRecipe(shape_plane_graph_t*& spr, ParamsManagerBase* pParams, RecipeAllocator* recipeAllocator = nullptr)
    : m_spr(spr), m_recipeOp(pParams, recipeAllocator)
    {
    }

    ScanRecipe(const shape_plane_graph_t*& spr, ParamsManagerBase* pParams, RecipeAllocator* recipeAllocator = nullptr)
    : ScanRecipe(const_cast<shape_plane_graph_t*&>(spr), pParams, recipeAllocator)
    {
    }

    void scan();

private:
    shape_plane_graph_t*& m_spr;
    RecipeOp              m_recipeOp;
};

class WriteToDisk
{
public:
    explicit WriteToDisk(ParamsManagerBase* pParams, RecipeAllocator* recipeAllocator)
    : m_pParams(pParams), m_recipeAllocator(recipeAllocator)
    {
    }

    template<typename T>
    void head(T* ptr);
    template<typename T>
    void single(T* element);
    template<typename T>
    void array(T* element, int n);

    void charArray(const char* s);

private:
    ParamsManagerBase* m_pParams;
    RecipeAllocator*   m_recipeAllocator;
};

class ReadFromDisk
{
public:
    explicit ReadFromDisk(ParamsManagerBase* pParams, RecipeAllocator* recipeAllocator)
    : m_pParams(pParams), m_recipeAllocator(recipeAllocator)
    {
    }

    template<typename T>
    void head(T*& ptr);
    template<typename T>
    void single(T*& element);
    template<typename T>
    void array(T*& element, int n);

    void charArray(const char*& s);

private:
    ParamsManagerBase* m_pParams;
    RecipeAllocator*   m_recipeAllocator;
};

using ScanRecipeWrite = ScanRecipe<WriteToDisk>;
using ScanRecipeRead  = ScanRecipe<ReadFromDisk>;
