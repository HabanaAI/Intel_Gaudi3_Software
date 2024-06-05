#include "data_layout_test_infra.h"
#include "node_factory.h"

const char* SynGaudiDataLayoutTest::pt_conv2D_in_layouts[]  = {"WHCN", "SRCK", "", "WHCN", ""};
const char* SynGaudiDataLayoutTest::pt_conv2D_out_layouts[] = {"WHCN"};
const char* SynGaudiDataLayoutTest::pt_conv3D_in_layouts[]  = {"WHDCN", "SRQCK", "", "WHDCN", ""};
const char* SynGaudiDataLayoutTest::pt_conv3D_out_layouts[] = {"WHDCN"};
const char* SynGaudiDataLayoutTest::pt_dedx_in_layouts[]    = {"WHCN", "SRCK", "WHCN"};
const char* SynGaudiDataLayoutTest::pt_dedx_out_layouts[]   = {"WHCN"};
const char* SynGaudiDataLayoutTest::pt_dedx3d_in_layouts[]  = {"WHDCN", "SRQCK", "WHDCN"};
const char* SynGaudiDataLayoutTest::pt_dedx3d_out_layouts[] = {"WHDCN"};
const char* SynGaudiDataLayoutTest::pt_dedw_in_layouts[]    = {"WHCN", "WHCN"};
const char* SynGaudiDataLayoutTest::pt_dedw_out_layouts[]   = {"SRCK"};
const char* SynGaudiDataLayoutTest::pt_dedw3d_in_layouts[]  = {"WHDCN", "WHDCN"};
const char* SynGaudiDataLayoutTest::pt_dedw3d_out_layouts[] = {"SRQCK"};

void SynGaudiDataLayoutTest::setPermutation(unsigned t, const gc::Permutation& perm)
{
    synTensorPermutation permutation;
    permutation.dims = perm.size();
    unsigned i       = 0;
    for (unsigned dim : perm.getValues())
    {
        permutation.permutation[i++] = dim;
    }
    synTensorSetPermutation(getTensorByIndex(t), &permutation);
}

SynGaudiDataLayoutTest::InOutLayouts SynGaudiDataLayoutTest::getNodePtLayout(const char* guid)
{
    if (guid == NodeFactory::convolutionNodeTypeName)
    {
        return std::make_pair(pt_conv2D_in_layouts, pt_conv2D_out_layouts);
    }
    else if (guid == NodeFactory::convolution3DNodeTypeName)
    {
        return std::make_pair(pt_conv3D_in_layouts, pt_conv3D_out_layouts);
    }
    else if (guid == NodeFactory::deDxNodeTypeName)
    {
        return std::make_pair(pt_dedx_in_layouts, pt_dedx_out_layouts);
    }
    else if (guid == NodeFactory::deDx3DNodeTypeName)
    {
        return std::make_pair(pt_dedx3d_in_layouts, pt_dedx3d_out_layouts);
    }
    else if (guid == NodeFactory::deDwNodeTypeName)
    {
        return std::make_pair(pt_dedw_in_layouts, pt_dedw_out_layouts);
    }
    else if (guid == NodeFactory::deDw3DNodeTypeName)
    {
        return std::make_pair(pt_dedw3d_in_layouts, pt_dedw3d_out_layouts);
    }
    else
    {
        return std::make_pair(nullptr, nullptr);
    }
}

void SynGaudiDataLayoutTest::addNodeWithLayouts(const char*   guid,
                                                TensorIndices inputTensorIndices,
                                                TensorIndices outputTensorIndices,
                                                void*         userParams,
                                                unsigned      paramsSize)
{
    InOutLayouts io_layouts = getNodePtLayout(guid);
    addNodeToGraph(guid,
                   inputTensorIndices,
                   outputTensorIndices,
                   userParams,
                   paramsSize,
                   nullptr,
                   0,
                   nullptr,
                   io_layouts.first,
                   io_layouts.second);
}

gc::Permutation SynGaudiDataLayoutTest::getPermutation(const char* name)
{
    synRecipeHandle recipeHandle = getRecipeHandle();
    recipe_t*       currRecipe   = recipeHandle->basicRecipeHandle.recipe;
    for (unsigned i = 0; i < currRecipe->persist_tensors_nr; i++)
    {
        const persist_tensor_info_t& t = currRecipe->tensors[i];
        if (std::strcmp(t.name, name) == 0)
        {
            gc::Permutation permutation(DimVector(t.permutation, t.permutation + t.dimensions));
            return permutation;
        }
    }
    HB_ASSERT(0, "tensor {} not found", name);
    return false;
}

bool SynGaudiDataLayoutTest::isPermuted(const char* name, const gc::Permutation& expected)
{
    return getPermutation(name) == expected;
}
