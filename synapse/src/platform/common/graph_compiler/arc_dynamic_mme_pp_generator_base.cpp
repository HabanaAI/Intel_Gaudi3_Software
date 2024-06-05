#include "arc_dynamic_mme_pp_generator.h"

namespace arc_platforms
{
int DynamicMMEPatchPointGeneratorBase::getTensorIndex(const TensorVector& tensorVector, const TensorPtr& tensor)
{
    int i = 0;

    for (const TensorPtr& curr : tensorVector)
    {
        if (curr == tensor)
        {
            return i;
        }
        i++;
    }
    return INDEX_NOT_APPLICABLE;
}

}  // namespace arc_platforms
