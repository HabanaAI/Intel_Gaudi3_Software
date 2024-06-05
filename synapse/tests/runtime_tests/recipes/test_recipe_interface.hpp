#pragma once

#include "test_sections_container.hpp"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "../infra/test_types.hpp"

#include "test_host_buffer_alloc.hpp"
#include "test_tensor_info.hpp"
#include "test_tensor_memory.hpp"

class TestTensorsContainer;
class TestEvent;

class TestRecipeInterface
{
public:
    virtual ~TestRecipeInterface() = default;

    synStatus         getConstSectionProperties(bool&              isConst,
                                                uint64_t&          sectionSize,
                                                uint64_t&          hostBuffer,
                                                const synSectionId sectionId) const;
    virtual void      getIsConstProperty(bool& isConst, const synSectionId sectionId) const                  = 0;
    virtual void      getSectionSizeProperty(uint64_t& sectionSize, const synSectionId sectionId) const      = 0;
    virtual synStatus getSectionHostBufferProperty(uint64_t& hostBuffer, const synSectionId sectionId) const = 0;

    virtual void                   recipeSerialize()        = 0;
    virtual bool                   recipeDeserialize()      = 0;
    virtual void                   generateRecipe()         = 0;
    virtual const synRecipeHandle& getRecipe() const        = 0;
    virtual uint64_t               getWorkspaceSize() const = 0;

    virtual void validateResults(const LaunchTensorMemory& rLaunchTensorMemory) const { return; }

    virtual void validateResults(const LaunchTensorMemory&    rLaunchTensorMemory,
                                 const std::vector<uint64_t>& rConstSectionsHostBuffers) const
    {
        return validateResults(rLaunchTensorMemory);
    }

    virtual uint64_t getTensorSizeInput(unsigned tensorIndex) const  = 0;
    virtual uint64_t getTensorSizeOutput(unsigned tensorIndex) const = 0;

    virtual uint64_t getDynamicTensorMaxDimSizeInput(unsigned tensorIndex, unsigned dimIndex) const  = 0;
    virtual uint64_t getDynamicTensorMaxDimSizeOutput(unsigned tensorIndex, unsigned dimIndex) const = 0;

    virtual const SectionInfoVec& getUniqueSectionsInfo() const = 0;
    virtual SectionMemoryVec&     getUniqueSectionsMemory()     = 0;

    virtual unsigned getTensorInfoVecSizeInput() const  = 0;
    virtual unsigned getTensorInfoVecSizeOutput() const = 0;
    virtual unsigned getTensorInfoVecSize() const       = 0;

    virtual const TensorInfoVec& getTensorInfoVecInputs() const  = 0;
    virtual const TensorInfoVec& getTensorInfoVecOutputs() const = 0;

    virtual const TensorInfo* getTensorInfoInput(unsigned tensorIndex) const     = 0;
    virtual const TensorInfo* getTensorInfoOutput(unsigned tensorIndex) const    = 0;
    virtual const TensorInfo* getTensorInfo(const std::string& tensorName) const = 0;

    virtual const std::vector<const char*> getTensorsName() const = 0;

    virtual bool isInputOnConstSection(unsigned tensorIndex) const  = 0;
    virtual bool isOutputOnConstSection(unsigned tensorIndex) const = 0;

    virtual void clearConstSectionHostBuffers(const std::vector<synSectionId>& constSectionIds) const = 0;

    virtual void getConstSectionInfo(uint64_t& sectionSize, void*& hostSectionData, synSectionId sectionId) const = 0;

    void createUniqueSections(TestSectionsContainer& sections, const synGraphHandle& graphHandle);

    static void createSection(synSectionHandle*    pSectionHandle,
                              const synGraphHandle graphHandle,
                              bool                 isPersist,
                              bool                 isConstSection);

    virtual unsigned getNumberInputConstTensors() const = 0;

    virtual void                         retrieveAmountOfExternalTensors(uint64_t& amountOfExternalTensors) = 0;
    virtual const std::vector<uint64_t>& retrieveTensorsId() const                                          = 0;
    virtual const std::vector<uint64_t>& retrieveOrderedTensorsId() const                                   = 0;
    // Maps a single event to a given launch-tensor
    virtual void mapEventToExternalTensor(TestEvent& rEvent, const synLaunchTensorInfoExt* launchTensorsInfo) const = 0;

    // minTensorSize - in case nullptr or tensor-type is not DATA_TENSOR_DYNAMIC it will not be used
    static void createTrainingTensor(TestTensorsContainer& tensors,
                                     unsigned              tensorIndex,
                                     const TensorInfo&     tensorInfo,
                                     bool                  isPersist,
                                     const std::string&    name,
                                     const synGraphHandle  graphHandle,
                                     synSectionHandle*     pGivenSectionHandle,
                                     void*                 hostBuffer);

    // TODO - remove this
    // minTensorSize - in case nullptr or tensor-type is not DATA_TENSOR_DYNAMIC it will not be used
    static void createTrainingTensor(TestTensorsContainer& tensors,
                                     unsigned              tensorIndex,
                                     TSize                 dims,
                                     synDataType           dataType,
                                     const TSize*          tensorSize,
                                     bool                  isPersist,
                                     const std::string&    name,
                                     const synGraphHandle  graphHandle,
                                     synSectionHandle*     pGivenSectionHandle,
                                     bool                  isConstSection,
                                     uint64_t              offset,
                                     void*                 hostBuffer,
                                     synTensorType         tensorType,
                                     const TSize*          minTensorSize);

    static void clearResourceFiles();

    unsigned getSectionID(TensorInfo& tensorInfo) const;

protected:
    virtual void _createGraphHandle()  = 0;
    virtual void _destroyGraphHandle() = 0;
};
