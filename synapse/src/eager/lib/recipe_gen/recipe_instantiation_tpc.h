
#pragma once

struct blob_t;

namespace eager_mode
{
struct TemplateOfEngine;
class DescGeneratorBase;
class RecipeHalBase;

// Assign actual values from tensor/non-tensor descriptors to execution blobs and memcpy
// work distribution FW context from structs to dynamic blob.
class TpcInstantiation
{
public:
    TpcInstantiation(const TemplateOfEngine&  tpcTemplate,
                     const DescGeneratorBase& descGenerator,
                     const RecipeHalBase&     recipeHal);
    void instantiateDynBlobs(blob_t& actualBlob);
    void instantiateExcBlobs(blob_t* actualBlobs);

private:
    const TemplateOfEngine&  m_template;
    const DescGeneratorBase& m_descGenerator;
    const RecipeHalBase&     m_recipeHal;
};

}  // namespace eager_mode
