/*****************************************************************************
 * Copyright (C) 2018 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Tzachi Cohen <tcohen@habana.ai>
 ******************************************************************************
 */

#ifndef _SYN_SINGLETON_INTERFACE_H_
#define _SYN_SINGLETON_INTERFACE_H_

#include <cstdint>
#include <vector>
#include <string>

#include "synapse_common_types.h"
#include "synapse_types.h"
#include "recipe.h"
#include "compiler_types.h"
#include "smf_callbacks.hpp"
#include "hlthunk.h"

#define SYN_INTERNAL_STREAM_FLAG (0x100)

#define VERIFY_ORIGINAL_IMPL(_ret_)                                                                                    \
    if (m_originalImpl == nullptr) return _ret_;
#define VERIFY_ORIGINAL_IMPL_VOID                                                                                      \
    if (m_originalImpl == nullptr) return;
#define VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED VERIFY_ORIGINAL_IMPL(synUnsupported)
#define VERIFY_ORIGINAL_IMPL_RET_NULL        VERIFY_ORIGINAL_IMPL(nullptr)

#define SYNAPSE_SINGLETON_INTERFACE_VERSION "1.14.0.2"

class synSingletonInterface
{
public:
    synSingletonInterface(synSingletonInterface* originalImpl) { m_originalImpl = originalImpl; }

    virtual ~synSingletonInterface() {}

    virtual synStatus initialize() = 0;

    virtual synStatus destroy() = 0;

    virtual synStatus acquireDevice(uint32_t*     pDeviceId,
                                    const char*   pciBus,
                                    synDeviceType deviceType = synDeviceTypeInvalid,
                                    synModuleId   moduleId   = INVALID_MODULE_ID)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->acquireDevice(pDeviceId, pciBus, deviceType, moduleId);
    }

    virtual synStatus releaseDevice(uint32_t deviceId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->releaseDevice(deviceId);
    }

    virtual uint16_t getNumOfAcquiredDevices()
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getNumOfAcquiredDevices();
    }

    virtual synStatus
    createGraph(synGraphHandle* pGraphHandle, synDeviceType deviceType, CompilationMode compilationMode)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createGraph(pGraphHandle, deviceType, compilationMode);
    }

    virtual synStatus destroyGraph(const synGraphHandle graphHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->destroyGraph(graphHandle);
    }

    virtual synStatus getDeviceDramMemoryInfo(uint32_t device, uint64_t& free, uint64_t& total) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceDramMemoryInfo(device, free, total);
    }

    virtual synStatus getGraphDeviceType(const synGraphHandle graphHandle, synDeviceType* deviceType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getGraphDeviceType(graphHandle, deviceType);
    }

    virtual synTensor createTensor(synTensorDescriptor* pDescriptor,
                                   synStatus*           pStatus,
                                   bool                 isOutput      = false,
                                   bool                 isInput       = false,
                                   bool                 isStaticParam = false)
    {
        VERIFY_ORIGINAL_IMPL_RET_NULL
        return m_originalImpl->createTensor(pDescriptor, pStatus, isOutput, isInput, isStaticParam);
    }

    virtual synStatus createTensor(const synTensorDescriptor* pDescriptor,
                                   synTensor*                 tensor,
                                   synSectionHandle           sectionHandle = nullptr,
                                   uint64_t                   sectionOffset = 0,
                                   bool                       isConst       = false)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createTensor(pDescriptor, tensor, sectionHandle, sectionOffset, isConst);
    }

    virtual synStatus createTensor(synTensor* tensor, synGraphHandle graph, synTensorType type, const char* tensorName)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createTensor(tensor, graph, type, tensorName);
    }

    virtual synStatus tensorAssignToSection(synTensor tensor, synSectionHandle section, uint64_t byteOffset)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorAssignToSection(tensor, section, byteOffset);
    }

    virtual synStatus
    tensorSetHostPtr(synTensor tensor, void* hostPtr, uint64_t size, synDataType dataType, bool copyBuffer)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetHostPtr(tensor, hostPtr, size, dataType, copyBuffer);
    }

    virtual synStatus
    tensorSetQuantizationData(synTensor tensor, synQuantizationProperty prop, void* propVal, uint64_t propSize)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetQuantizationData(tensor, prop, propVal, propSize);
    }

    virtual synStatus
    tensorSetGeometry(synTensor tensor, const synTensorGeometry* geometry, synGeometryType geometryType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetGeometry(tensor, geometry, geometryType);
    }

    virtual synStatus tensorGetDeviceLayout(const synTensor tensor, synTensorDeviceLayout* layout)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetDeviceLayout(tensor, layout);
    }

    virtual synStatus tensorSetDeviceLayout(synTensor tensor, const synTensorDeviceLayout* layout)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetDeviceLayout(tensor, layout);
    }

    virtual synStatus
    tensorGetGeometry(const synTensor tensor, synTensorGeometry* geometry, synGeometryType geometryType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetGeometry(tensor, geometry, geometryType);
    }

    virtual synStatus tensorGetName(const synTensor tensor, const uint64_t size, char* name)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetName(tensor, size, name);
    }

    virtual synStatus tensorGetType(synTensor tensor, synTensorType* type)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetType(tensor, type);
    }

    virtual synStatus setTensorDeviceAddr(synTensor tensor, uint64_t addr)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setTensorDeviceAddr(tensor, addr);
    }

    virtual synStatus setGraphDeviceAddress(int32_t deviceId, uint64_t size, uint64_t buffer)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setGraphDeviceAddress(deviceId, size, buffer);
    }

    virtual synStatus destroyTensor(synTensor tensor)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->destroyTensor(tensor);
    }

    virtual synStatus compileGraph(synRecipeHandle*     pRecipeHandle,
                                   const synGraphHandle graphHandle,
                                   const char*          fileName,
                                   const char*          buildLog)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->compileGraph(pRecipeHandle, graphHandle, fileName, buildLog, 0, 0);
    }

    virtual synStatus compileGraph(synRecipeHandle*     pRecipeHandle,
                                   const synGraphHandle graphHandle,
                                   const char*          fileName,
                                   const char*          buildLog,
                                   uint64_t             size,
                                   uint64_t             buffer)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->compileGraph(pRecipeHandle, graphHandle, fileName, buildLog);
    }

    virtual synStatus createGenericNode(const synGraphHandle graphHandle,
                                        const synTensor*     pInputsTensorList,
                                        const synTensor*     outputs,
                                        const uint32_t       sizeInputs,
                                        const uint32_t       sizeOutputs,
                                        const void*          userParams,
                                        const unsigned       paramsSize,
                                        const char*          guid,
                                        const char**         inputLayouts,
                                        const char**         outputLayouts,
                                        const std::string&   name = "")
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createGenericNode(graphHandle,
                                                 pInputsTensorList,
                                                 outputs,
                                                 sizeInputs,
                                                 sizeOutputs,
                                                 userParams,
                                                 paramsSize,
                                                 guid,
                                                 inputLayouts,
                                                 outputLayouts,
                                                 name);
    }

    virtual synStatus createGenericNodeWithId(const synGraphHandle graphHandle,
                                              const synTensor*     pInputsTensorList,
                                              const synTensor*     outputs,
                                              const uint32_t       sizeInputs,
                                              const uint32_t       sizeOutputs,
                                              const void*          userParams,
                                              const unsigned       paramsSize,
                                              const char*          guid,
                                              const char**         inputLayouts,
                                              const char**         outputLayouts,
                                              const std::string&   name         = "",
                                              synNodeId*           nodeUniqueId = nullptr)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createGenericNodeWithId(graphHandle,
                                                       pInputsTensorList,
                                                       outputs,
                                                       sizeInputs,
                                                       sizeOutputs,
                                                       userParams,
                                                       paramsSize,
                                                       guid,
                                                       inputLayouts,
                                                       outputLayouts,
                                                       name,
                                                       nodeUniqueId);
    }

    virtual synStatus
    nodeTypeSetUserPrecision(const synGraphHandle graphHandle, const char* guid, synDataType nodePrecision)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->nodeTypeSetUserPrecision(graphHandle, guid, nodePrecision);
    }

    virtual synStatus allocateDeviceMemory(unsigned     deviceId,
                                           uint64_t     size,
                                           uint32_t     flags,
                                           void**       buffer,
                                           uint64_t     reqVAAddress = 0,
                                           uint64_t*    deviceVA     = nullptr)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->allocateDeviceMemory(deviceId, size, flags, buffer, reqVAAddress, deviceVA);
    }

    virtual synStatus deallocateDeviceMemory(unsigned deviceId, void* pBuffer, uint32_t flags = synMemFlags::synMemHost)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->deallocateDeviceMemory(deviceId, pBuffer, flags);
    }

    virtual synStatus mapBufferToDevice(unsigned deviceId, uint64_t size, void* buffer, uint64_t reqVAAddress = 0)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->mapBufferToDevice(deviceId, size, buffer, reqVAAddress);
    }

    virtual synStatus unmapBufferFromDevice(unsigned deviceId, void* buffer)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->unmapBufferFromDevice(deviceId, buffer);
    }

    virtual synStatus tensorRetrieveMetadatasInfosByName(const synRecipeHandle pRecipeHandle,
                                                         const uint32_t        numOfTensors,
                                                         TensorMetadataInfo*   tensorsMetadataInfo) const
    {
        return synUnsupported;
    }

    virtual synStatus tensorRetrievePersistentAmount(const synRecipeHandle pRecipeHandle, uint32_t& numOfTensors) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorRetrievePersistentAmount(pRecipeHandle, numOfTensors);
    }

    virtual synStatus tensorRetrieveNames(const synRecipeHandle pRecipeHandle,
                                          char                  tensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                                          const uint32_t        numOfTensors) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorRetrieveNames(pRecipeHandle, tensorsName, numOfTensors);
    }

    virtual synStatus tensorRetrieveLaunchAmount(const synRecipeHandle pRecipeHandle, uint32_t& numOfTensors) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorRetrieveLaunchAmount(pRecipeHandle, numOfTensors);
    }

    virtual synStatus tensorRetrieveLaunchIds(const synRecipeHandle pRecipeHandle,
                                              uint64_t*             tensorsIds,
                                              const uint32_t        numOfTensors) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorRetrieveLaunchIds(pRecipeHandle, tensorsIds, numOfTensors);
    }

    virtual synStatus tensorRetrieveLaunchInfoById(const synRecipeHandle         pRecipeHandle,
                                                   const uint32_t                numOfTensors,
                                                   synRetrievedLaunchTensorInfo* tensorsLaunchInfo) const
    {
        return synUnsupported;
    }

    virtual synStatus sectionCreate(synSectionHandle* phSection, uint64_t sectionDescriptor, const synGraphHandle graph)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionCreate(phSection, sectionDescriptor, graph);
    }

    virtual synStatus sectionDestroy(synSectionHandle hSection)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionDestroy(hSection);
    }

    virtual synStatus sectionGroupSet(synSectionHandle hSection, uint64_t group) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionGroupSet(hSection, group);
    }
    virtual synStatus sectionGroupGet(synSectionHandle hSection, uint64_t* group) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionGroupGet(hSection, group);
    }
    virtual synStatus sectionPersistentSet(synSectionHandle hSection, bool isPersistent) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionPersistentSet(hSection, isPersistent);
    }
    virtual synStatus sectionPersistentGet(synSectionHandle hSection, bool* isPersistent) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionPersistentGet(hSection, isPersistent);
    }
    virtual synStatus sectionRMWSet(synSectionHandle hSection, bool isRMW) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionRMWSet(hSection, isRMW);
    }
    virtual synStatus sectionRMWGet(synSectionHandle hSection, bool* isRMW) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionRMWGet(hSection, isRMW);
    }
    virtual synStatus sectionSetDeviceAddress(synSectionHandle hSection, uint64_t deviceAddress) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionSetDeviceAddress(hSection, deviceAddress);
    }
    virtual synStatus sectionGetDeviceAddress(synSectionHandle hSection, uint64_t* deviceAddress) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionGetDeviceAddress(hSection, deviceAddress);
    }

    virtual synStatus recipeDestroy(synRecipeHandle hRecipe)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->recipeDestroy(hRecipe);
    }

    virtual synStatus getTopologyWorkspaceSize(uint64_t* pWorkspaceSize, const synRecipeHandle recipeHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getTopologyWorkspaceSize(pWorkspaceSize, recipeHandle);
    }

    // Deprecated API, can be reused as another API. keeping function signature for ABI compatibility.
    virtual synStatus
    createStream(synStreamHandle* pStreamHandle, const uint32_t deviceId, uint32_t streamType, const unsigned int flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createStream(pStreamHandle, deviceId, streamType, flags);
    }

    virtual synStatus destroyStream(synStreamHandle streamHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->destroyStream(streamHandle);
    }

    virtual synStatus createEvent(synEventHandle* pEventHandle, const uint32_t deviceId, const unsigned int flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createEvent(pEventHandle, deviceId, flags);
    }

    virtual synStatus destroyEvent(const synEventHandle eventHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->destroyEvent(eventHandle);
    }

    virtual synStatus synchronizeStream(const synStreamHandle streamHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->synchronizeStream(streamHandle);
    }

    virtual synStatus synchronizeAllStreams(const uint32_t deviceId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->synchronizeAllStreams(deviceId);
    }

    virtual synStatus synchronizeEvent(const synEventHandle eventHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->synchronizeEvent(eventHandle);
    }

    virtual synStatus
    eventElapsedTime(uint64_t* pNanoseconds, const synEventHandle eventHandleStart, const synEventHandle eventHandleEnd)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->eventElapsedTime(pNanoseconds, eventHandleStart, eventHandleEnd);
    }

    virtual synStatus
    streamWaitEvent(const synStreamHandle streamHandle, const synEventHandle eventHandle, const unsigned int flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->streamWaitEvent(streamHandle, eventHandle, flags);
    }

    virtual synStatus eventRecord(const synEventHandle eventHandle, const synStreamHandle streamHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->eventRecord(eventHandle, streamHandle);
    }

    virtual synStatus eventQuery(const synEventHandle eventHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->eventQuery(eventHandle);
    }

    virtual synStatus streamQuery(const synStreamHandle streamHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->streamQuery(streamHandle);
    }

    virtual synStatus recipeSerialize(const synRecipeHandle recipeHandle, const char* recipeFileName)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->recipeSerialize(recipeHandle, recipeFileName);
    }

    virtual synStatus recipeDeSerialize(synRecipeHandle* pRecipeHandle, const char* recipeFileName)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->recipeDeSerialize(pRecipeHandle, recipeFileName);
    }

    virtual synStatus recipeGetAttribute(uint64_t*                 retVal,
                                         const synRecipeAttribute* recipeAttr,
                                         const unsigned            querySize,
                                         const synRecipeHandle     recipeHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->recipeGetAttribute(retVal, recipeAttr, querySize, recipeHandle);
    }

    // Should be deprecated
    virtual synStatus enqueue(const synStreamHandle      streamHandle,
                              const synLaunchTensorInfo* enqueueInputTensorsInfo,
                              const uint32_t             inputSize,
                              const synLaunchTensorInfo* enqueueOutputTensorsInfo,
                              const uint32_t             outputSize,
                              uint64_t                   pWorkspace,
                              const synRecipeHandle      pRecipeHandle,
                              uint32_t                   flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->enqueue(streamHandle,
                                       enqueueInputTensorsInfo,
                                       inputSize,
                                       enqueueOutputTensorsInfo,
                                       outputSize,
                                       pWorkspace,
                                       pRecipeHandle,
                                       flags);
    }

    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueInputTensorsInfo,
                              const uint32_t                inputSize,
                              const synLaunchTensorInfoExt* enqueueOutputTensorsInfo,
                              const uint32_t                outputSize,
                              uint64_t                      pWorkspace,
                              const synRecipeHandle         pRecipeHandle,
                              uint32_t                      flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->enqueue(streamHandle,
                                       enqueueInputTensorsInfo,
                                       inputSize,
                                       enqueueOutputTensorsInfo,
                                       outputSize,
                                       pWorkspace,
                                       pRecipeHandle,
                                       flags);
    }

    // Should be deprecated
    virtual synStatus enqueue(const synStreamHandle      streamHandle,
                              const synLaunchTensorInfo* enqueueTensorsInfo,
                              const uint32_t             enqueueTensorsAmount,
                              uint64_t                   pWorkspace,
                              const synRecipeHandle      pRecipeHandle,
                              uint32_t                   flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl
            ->enqueue(streamHandle, enqueueTensorsInfo, enqueueTensorsAmount, pWorkspace, pRecipeHandle, flags);
    }

    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueTensorsInfo,
                              const uint32_t                enqueueTensorsAmount,
                              uint64_t                      pWorkspace,
                              const synRecipeHandle         pRecipeHandle,
                              uint32_t                      flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl
            ->enqueue(streamHandle, enqueueTensorsInfo, enqueueTensorsAmount, pWorkspace, pRecipeHandle, flags);
    }

    virtual synStatus deviceGetCount(uint32_t* pCount)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->deviceGetCount(pCount);
    }

    virtual synStatus deviceGetCountByDeviceType(uint32_t* pCount, const synDeviceType deviceType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->deviceGetCountByDeviceType(pCount, deviceType);
    }

    virtual synStatus deviceCount(uint32_t count[synDeviceTypeSize])
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->deviceCount(count);
    }

    virtual synStatus deviceGetPCIBusId(char* pPciBusId, const int len, const synDeviceId deviceId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->deviceGetPCIBusId(pPciBusId, len, deviceId);
    }

    virtual synStatus
    writeI2cReg(uint32_t deviceId, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t value)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->writeI2cReg(deviceId, i2cBus, i2cAddress, regAddress, value);
    }

    virtual synStatus
    readI2cReg(uint32_t deviceId, uint32_t i2cBus, uint32_t i2cAddress, uint32_t regAddress, uint32_t* pValue)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->readI2cReg(deviceId, i2cBus, i2cAddress, regAddress, pValue);
    }

    virtual synStatus setLedState(uint32_t deviceId, uint32_t ledId, bool state)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setLedState(deviceId, ledId, state);
    }

    virtual synStatus setFrequency(uint32_t deviceId, uint32_t pllId, uint32_t frequency)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setFrequency(deviceId, pllId, frequency);
    }

    virtual synStatus getFrequency(uint32_t deviceId, uint32_t pllId, uint32_t* pFrequency)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getFrequency(deviceId, pllId, pFrequency);
    }

    virtual synStatus memcpyAsync(const synStreamHandle streamHandle,
                                  const uint64_t*       pSrc,
                                  const uint64_t*       pSize,
                                  const uint64_t*       pDst,
                                  const synDmaDir       direction,
                                  const uint64_t        numCopies = 1)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->memcpyAsync(streamHandle, pSrc, pSize, pDst, direction, numCopies);
    }

    virtual synStatus memsetAsync(const synStreamHandle streamHandle,
                                  uint64_t              pDeviceMem,
                                  const uint32_t        value,
                                  const size_t          numOfElements,
                                  const size_t          elementSize)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->memsetAsync(streamHandle, pDeviceMem, value, numOfElements, elementSize);
    }

    virtual synStatus getDeviceName(char* pName, const int len, const synDeviceId deviceId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceName(pName, len, deviceId);
    }

    virtual synStatus getDeviceId(const synStreamHandle streamHandle, synDeviceId& deviceId) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceId(streamHandle, deviceId);
    }

    virtual synStatus getDeviceInfo(unsigned deviceId, synDeviceInfo* pDeviceInfo) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceInfo(deviceId, pDeviceInfo);
    }

    virtual synStatus getDeviceAttribute(const synDeviceId         deviceId,
                                         const synDeviceAttribute* deviceAttr,
                                         const unsigned            querySize,
                                         uint64_t*                 retVal) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceAttribute(deviceId, deviceAttr, querySize, retVal);
    }

    virtual synStatus setCfg(const char* cfgName, const char* cfgValue)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setCfg(cfgName, cfgValue);
    }

    virtual synStatus getCfg(const char* cfgName, char* cfgValue, uint64_t size)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getCfg(cfgName, cfgValue, size);
    }

    virtual synStatus profile(unsigned deviceId, hl_debug_args* debugParams)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profile(deviceId, debugParams);
    }

    virtual synStatus profilerStart(synTraceType type, const synDeviceId deviceId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profilerStart(type, deviceId);
    }

    virtual synStatus profilerStop(synTraceType type, const synDeviceId deviceId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profilerStop(type, deviceId);
    }

    virtual synStatus profilerGetTrace(synTraceType      type,
                                       const synDeviceId deviceId,
                                       synTraceFormat    format,
                                       void*             buffer,
                                       size_t*           size,
                                       size_t*           numEntries)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profilerGetTrace(type, deviceId, format, buffer, size, numEntries);
    }

    virtual synStatus deviceGetFd(int* pFd, const synDeviceId deviceId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->deviceGetFd(pFd, deviceId);
    }

    virtual synStatus getRecipeDebugInfo(synRecipeHandle recipe, const debug_info_t** recipeDebugInfo)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getRecipeDebugInfo(recipe, recipeDebugInfo);
    }

    virtual synStatus getRecipeProgramDataBlobs(synRecipeHandle             recipe,
                                                const program_data_blob_t** program_data_blobs,
                                                size_t*                     program_data_blobs_nr)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getRecipeProgramDataBlobs(recipe, program_data_blobs, program_data_blobs_nr);
    }

    virtual synStatus getClockSyncInfo(unsigned deviceId, hlthunk_time_sync_info* infoOut)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getClockSyncInfo(deviceId, infoOut);
    }

    virtual synStatus getPllFrequency(unsigned deviceId, uint32_t index, struct hlthunk_pll_frequency_info* freqOut)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getPllFrequency(deviceId, index, freqOut);
    }

    virtual synStatus createControlDependency(const synGraphHandle graphHandle,
                                              const synNodeId*     pBlockingNodesIdList,
                                              const synNodeId*     pBlockedNodesIdList,
                                              const uint32_t       numberblocking,
                                              const uint32_t       numberblocked)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createControlDependency(graphHandle,
                                                       pBlockingNodesIdList,
                                                       pBlockedNodesIdList,
                                                       numberblocking,
                                                       numberblocked);
    }

    virtual synStatus getUniqueNodeId(synNodeId& nodeId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getUniqueNodeId(nodeId);
    }

    virtual synStatus
    setOriginalComplexNode(synGraphHandle graphHandle, synNodeId nodeId, synNodeId originalComplexNodeId)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setOriginalComplexNode(graphHandle, nodeId, originalComplexNodeId);
    }

    virtual uint64_t profilerGetCurrentTimeNs()
    {
        VERIFY_ORIGINAL_IMPL(0);
        return m_originalImpl->profilerGetCurrentTimeNs();
    }

    virtual synStatus profileInternalFunction(const char* funcName, uint64_t startTime)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profileInternalFunction(funcName, startTime);
    }

    virtual synStatus dumpProfilerJson()
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->dumpProfilerJson();
    }

    virtual synStatus tensorGetSection(synTensor tensor, synSectionHandle* section, uint64_t* byteOffset)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetSection(tensor, section, byteOffset);
    }

    virtual synStatus tensorGetHostPtr(synTensor tensor, void** hostPtr, uint64_t* size, synDataType* dataType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetHostPtr(tensor, hostPtr, size, dataType);
    }

    virtual synStatus
    tensorGetQuantizationData(synTensor tensor, synQuantizationProperty prop, void* propVal, uint64_t propSize)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetQuantizationData(tensor, prop, propVal, propSize);
    }

    virtual synStatus tensorRetrieveIds(const synRecipeHandle pRecipeHandle,
                                        const char**          tensorNames,
                                        uint64_t*             tensorIds,
                                        const uint32_t        numOfTensors)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorRetrieveIds(pRecipeHandle, tensorNames, tensorIds, numOfTensors);
    }

    virtual synStatus getDeviceTypeAttribute(const synDeviceType       deviceType,
                                             const synDeviceAttribute* deviceAttr,
                                             const unsigned            querySize,
                                             uint64_t*                 retVal) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceTypeAttribute(deviceType, deviceAttr, querySize, retVal);
    }

    virtual synStatus supportsComplexGuid()
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->supportsComplexGuid();
    }

private:
    synSingletonInterface(synSingletonInterface const&) = delete;
    void operator=(synSingletonInterface const&) = delete;

protected:
    synSingletonInterface* m_originalImpl;  // Store a pointer to the synapse synSingleton implementation

public:
    virtual synStatus tensorGetDeviceFullLayout(const synTensor tensor, synTensorDeviceFullLayout* layout)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetDeviceFullLayout(tensor, layout);
    }

    virtual synStatus tensorSetDeviceFullLayout(synTensor tensor, const synTensorDeviceFullLayout* layout)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetDeviceFullLayout(tensor, layout);
    }

    virtual synStatus tensorSetIsExternal(synTensor tensor, bool isExternal)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetIsExternal(tensor, isExternal);
    }

    virtual synStatus tensorGetIsExternal(const synTensor tensor, bool* isExternal)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetIsExternal(tensor, isExternal);
    }

    virtual synStatus enqueue(const synStreamHandle         streamHandle,
                              const synLaunchTensorInfoExt* enqueueTensorsInfo,
                              const uint32_t                enqueueTensorsAmount,
                              uint64_t                      pWorkspace,
                              const synRecipeHandle         pRecipeHandle,
                              synEventHandle*               eventHandleList,
                              uint32_t                      numberOfEvents,
                              uint32_t                      flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->enqueue(streamHandle,
                                       enqueueTensorsInfo,
                                       enqueueTensorsAmount,
                                       pWorkspace,
                                       pRecipeHandle,
                                       eventHandleList,
                                       numberOfEvents,
                                       flags);
    }

    virtual synStatus eventMapTensor(synEventHandle*               eventHandle,
                                     size_t                        numOfEvents,
                                     const synLaunchTensorInfoExt* launchTensorsInfo,
                                     const synRecipeHandle         recipeHandle) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->eventMapTensor(eventHandle, numOfEvents, launchTensorsInfo, recipeHandle);
    }

    virtual synStatus eventMapTensorBase(synEventHandle*            eventHandle,
                                         size_t                     numOfEvents,
                                         const synLaunchTensorInfo* launchTensorsInfo,
                                         const synRecipeHandle      recipeHandle) const
    {
        return synUnsupported;
    }

    virtual synStatus externalTensorsExtractExecutionOrder(const synRecipeHandle recipeHandle,
                                                           uint32_t              numOfEvents,
                                                           uint64_t*             tensorIds) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->externalTensorsExtractExecutionOrder(recipeHandle, numOfEvents, tensorIds);
    }

    virtual synStatus getTPCLibraryVersionSize(uint32_t* size)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getTPCLibraryVersionSize(size);
    }

    virtual synStatus getTPCLibraryVersions(const char** libs, uint32_t* versions)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getTPCLibraryVersions(libs, versions);
    }

    virtual synStatus enqueueWithExternalEvents(const synStreamHandle      streamHandle,
                                                const synLaunchTensorInfo* enqueueTensorsInfo,
                                                const uint32_t             enqueueTensorsAmount,
                                                uint64_t                   pWorkspace,
                                                const synRecipeHandle      pRecipeHandle,
                                                synEventHandle*            eventHandleList,
                                                uint32_t                   numberOfEvents,
                                                uint32_t                   flags)
    {
        return synUnsupported;
    }

    virtual synStatus tensorSetAllowPermutation(synTensor tensor, int8_t allow)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetAllowPermutation(tensor, allow);
    }

    virtual synStatus tensorGetAllowPermutation(synTensor tensor, int8_t* allow)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetAllowPermutation(tensor, allow);
    }

    virtual synStatus tensorSetPermutation(synTensor tensor, const synTensorPermutation* permutation)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetPermutation(tensor, permutation);
    }

    virtual synStatus tensorGetPermutation(const synTensor tensor, synTensorPermutation* permutation)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetPermutation(tensor, permutation);
    }

    virtual synStatus tensorRetrieveMultiLaunchInfoById(const synRecipeHandle         pRecipeHandle,
                                                        const uint64_t                tensorId,
                                                        synRetrievedLaunchTensorInfo* tensorLaunchInfo,
                                                        uint32_t*                     numLaunchInfos) const
    {
        return synUnsupported;
    }

    virtual synStatus
    nodeSetDeterministic(const synGraphHandle graphHandle, const synNodeId nodeId, const bool useDeterministic)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->nodeSetDeterministic(graphHandle, nodeId, useDeterministic);
    }

    virtual synStatus
    nodeGetDeterministic(const synGraphHandle graphHandle, const synNodeId nodeId, bool* pUseDeterministic)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->nodeGetDeterministic(graphHandle, nodeId, pUseDeterministic);
    }

    virtual synStatus sectionConstSet(synSectionHandle hSection, bool isConst) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionConstSet(hSection, isConst);
    }

    virtual synStatus sectionConstGet(synSectionHandle hSection, bool* isConst) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionConstGet(hSection, isConst);
    }

    virtual synStatus sectionGetProp(const synRecipeHandle  pRecipeHandle,
                                     const synSectionId     sectionId,
                                     const synSectionProp   prop,
                                     uint64_t*              propertyPtr) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionGetProp(pRecipeHandle, sectionId, prop, propertyPtr);
    }

    // Function retained for compatibility with old versions of synapse profiler
    virtual synStatus getDynamicShapesTensorInfoArray(synStreamHandle             streamHandle,
                                                      synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const
    {
        return synSuccess;
    }

    virtual synStatus
    nodeSetRoundingMode(const synGraphHandle graphHandle, const synNodeId nodeId, const synRoundingMode roundingMode)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->nodeSetRoundingMode(graphHandle, nodeId, roundingMode);
    }

    virtual synStatus
    nodeGetRoundingMode(const synGraphHandle graphHandle, const synNodeId nodeId, synRoundingMode* pRoundingMode)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->nodeGetRoundingMode(graphHandle, nodeId, pRoundingMode);
    }

    virtual synStatus getRecipeSyncScheme(const synRecipeHandle recipe, const debug_sync_scheme_t** recipeSyncScheme)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getRecipeSyncScheme(recipe, recipeSyncScheme);
    }

    virtual synStatus createStreamEx(synStreamHandle*   pStreamHandle,
                                     const synDeviceId  deviceId,
                                     synStreamPriority  priority,
                                     const unsigned int flags)
    {
        return synUnsupported;
    }

    virtual synStatus getStreamPriority(const synStreamHandle streamHandle, synStreamPriority* priority)
    {
        return synUnsupported;
    }

    virtual synStatus
    getDevicePriorityRange(const synDeviceId deviceId, synStreamPriority* minPriority, synStreamPriority* maxPriority)
    {
        return synUnsupported;
    }

    virtual synStatus tensorSetGeometryExt(synTensor                   tensor,
                                           const synTensorGeometryExt* geometry,
                                           synGeometryType             geometryType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetGeometryExt(tensor, geometry, geometryType);
    }

    virtual synStatus tensorGetGeometryExt(const synTensor         tensor,
                                           synTensorGeometryExt*   geometry,
                                           synGeometryType         geometryType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetGeometryExt(tensor, geometry, geometryType);
    }

    virtual synStatus tensorSetDeviceDataType(synTensor   tensor,
                                              synDataType deviceDataType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetDeviceDataType(tensor, deviceDataType);
    }

    virtual synStatus tensorGetDeviceDataType(synTensor   tensor,
                                              synDataType* deviceDataType)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorGetDeviceDataType(tensor, deviceDataType);
    }

    virtual synStatus duplicateGraph(synGraphHandle      graphHandle,
                                     synGraphHandle*     newGraphHandle,
                                     synTensorHandleMap* tensorsMap,
                                     uint32_t*           numTensors,
                                     synNodeHandleMap*   nodesMap,
                                     uint32_t*           numNodes)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->duplicateGraph(graphHandle, newGraphHandle, tensorsMap, numTensors, nodesMap, numNodes);
    }

    virtual synStatus setStreamAffinity(const synDeviceId deviceId, const synStreamHandle pStreamHandle, uint64_t streamAffinityMask)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setStreamAffinity(deviceId, pStreamHandle, streamAffinityMask);
    }

    virtual synStatus getStreamAffinity(const synDeviceId deviceId, const synStreamHandle pStreamHandle, uint64_t* streamAffinityMask)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getStreamAffinity(deviceId, pStreamHandle, streamAffinityMask);
    }

    virtual synStatus getDeviceAffinityMaskRange(const synDeviceId  deviceId,
                                                 uint64_t*          deviceAffinityMaskRange)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceAffinityMaskRange(deviceId, deviceAffinityMaskRange);
    }

    virtual synStatus nodeSetParams(const synGraphHandle graphHandle,
                                    const synNodeId      nodeId,
                                    const void*          userParams,
                                    const unsigned       paramsSize)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->nodeSetParams(graphHandle, nodeId, userParams, paramsSize);
    }

    virtual synStatus
    nodeGetParams(const synGraphHandle graphHandle, const synNodeId nodeId, void* userParams, unsigned* paramsSize)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->nodeGetParams(graphHandle, nodeId, userParams, paramsSize);
    }

    virtual synStatus tensorRetrieveMetadatasInfosByNameExt(const synRecipeHandle  pRecipeHandle,
                                                            const uint32_t         numOfTensors,
                                                            TensorMetadataInfoExt* tensorsMetadataInfo) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorRetrieveMetadatasInfosByNameExt(pRecipeHandle, numOfTensors, tensorsMetadataInfo);
    }

    virtual synStatus tensorRetrieveLaunchInfoByIdExt(const synRecipeHandle            pRecipeHandle,
                                                      const uint32_t                   numOfTensors,
                                                      synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorRetrieveLaunchInfoByIdExt(pRecipeHandle, numOfTensors, tensorsLaunchInfo);
    }

    virtual synStatus eventMapTensorBaseExt(synEventHandle*               eventHandle,
                                            size_t                        numOfEvents,
                                            const synLaunchTensorInfoExt* launchTensorsInfo,
                                            const synRecipeHandle         recipeHandle) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->eventMapTensorBaseExt(eventHandle, numOfEvents, launchTensorsInfo, recipeHandle);
    }

    virtual synStatus enqueueWithExternalEventsExt(const synStreamHandle         streamHandle,
                                                   const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                                   const uint32_t                enqueueTensorsAmount,
                                                   uint64_t                      pWorkspace,
                                                   const synRecipeHandle         pRecipeHandle,
                                                   synEventHandle*               eventHandleList,
                                                   uint32_t                      numberOfEvents,
                                                   uint32_t                      flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->enqueueWithExternalEventsExt(streamHandle,
                                                            enqueueTensorsInfo,
                                                            enqueueTensorsAmount,
                                                            pWorkspace,
                                                            pRecipeHandle,
                                                            eventHandleList,
                                                            numberOfEvents,
                                                            flags);
    }

    // Deprecated API, can be reused as another API. keeping function signature for ABI compatibility.
    virtual synStatus profilerGetTrace2(synTraceType      type,
                                        const synDeviceId deviceId,
                                        synTraceFormat    format,
                                        void*             buffer,
                                        size_t*           size,
                                        size_t*           numEntries)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profilerGetTrace2(type, deviceId, format, buffer, size, numEntries);
    }

    virtual synStatus createStream(synStreamHandle* pStreamHandle, const uint32_t deviceId, const unsigned int flags)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->createStream(pStreamHandle, deviceId, flags);
    }

    virtual synStatus setSmfCallbacks(smf_callbacks_t* callbacks)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->setSmfCallbacks(callbacks);
    }

    /*DEPRECATED*/
    virtual synStatus
    graphSetAttribute(synGraphHandle           graphHandle,
                      const synGraphAttribute* attributes,
                      const uint64_t*          values,
                      const uint32_t           size)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->graphSetAttribute(graphHandle, attributes, values, size);
    }

    /*DEPRECATED*/
    virtual synStatus
    graphGetAttribute(synGraphHandle           graphHandle,
                      const synGraphAttribute* attributes,
                      uint64_t*                values,
                      const uint32_t           size)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->graphGetAttribute(graphHandle, attributes, values, size);
    }

    virtual synStatus deviceGetModuleIds(uint32_t *pDeviceModuleIds, uint32_t*  size)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->deviceGetModuleIds(pDeviceModuleIds, size);
    }

    virtual synStatus profileInternalFunctionWithArgs(const char* funcName, uint64_t startTime, const char** args, size_t argsSize)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profileInternalFunctionWithArgs(funcName, startTime, args, argsSize);
    }

    virtual synStatus profilerAddCustomEvent(const char* funcName, uint64_t startTime, uint64_t endTime, const char** args, size_t argsSize)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profilerAddCustomEvent(funcName, startTime, endTime, args, argsSize);
    }

    virtual synStatus profilerQueryRequiredMemory(const synDeviceId deviceId, uint32_t* bytesRequired)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profilerQueryRequiredMemory(deviceId, bytesRequired);
    }

    virtual synStatus profilerSetUserBuffer(const synDeviceId deviceId, void* userBuffer)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profilerSetUserBuffer(deviceId, userBuffer);
    }

    virtual synStatus getDeviceAttributesByModuleId(const synModuleId         moduleId,
                                                    const synDeviceAttribute* deviceAttr,
                                                    const unsigned            querySize,
                                                    uint64_t*                 retVal) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceAttributesByModuleId(moduleId, deviceAttr, querySize, retVal);
    }

    virtual synStatus setHostProfilerArg(const std::vector<synTraceEventArg>& keyValArgs)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        // Adds host profiling argument to the current upper syn_singleton API call in this threads stack
        return m_originalImpl->setHostProfilerArg(keyValArgs);
    }

    virtual synStatus getDeviceNextStreamAffinity(const synDeviceId deviceId, uint64_t* nextDeviceAffinity)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceNextStreamAffinity(deviceId, nextDeviceAffinity);
    }

    virtual synStatus sectionsClearHostBuffer( synRecipeHandle     recipeHandle,
                                               const synSectionId* sectionIds,
                                               size_t              numOfSections )
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->sectionsClearHostBuffer(recipeHandle, sectionIds, numOfSections);
    }

    virtual synStatus tensorSetSectionOffset(synTensor tensor, uint64_t byteOffset)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->tensorSetSectionOffset(tensor, byteOffset);
    }

    virtual synStatus
    graphSetAttributes(synGraphHandle              graphHandle,
                       const synGraphAttribute*    attributes,
                       const synGraphAttributeVal* values,
                       const uint32_t              size)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->graphSetAttributes(graphHandle, attributes, values, size);
    }

    virtual synStatus
    graphGetAttributes(synGraphHandle           graphHandle,
                       const synGraphAttribute* attributes,
                       synGraphAttributeVal*    values,
                       const uint32_t           size)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->graphGetAttributes(graphHandle, attributes, values, size);
    }

    virtual synStatus inferGraphShapes(const synGraphHandle graphHandle)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->inferGraphShapes(graphHandle);
    }

    virtual synStatus getClockSyncPerDieInfo(unsigned deviceId, uint32_t dieIndex,  hlthunk_time_sync_info* infoOut)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getClockSyncPerDieInfo(deviceId, dieIndex, infoOut);
    }

    virtual synStatus profileInternalFunctionWithArgsAndThread(const char*  funcName,
                                                               uint64_t     startTime,
                                                               const char** args,
                                                               size_t       argsSize,
                                                               const char*  threadName)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->profileInternalFunctionWithArgsAndThread(funcName, startTime, args, argsSize, threadName);
    }

    virtual synStatus getDeviceInfo(unsigned deviceId, synDeviceInfoV2* pDeviceInfo) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDeviceInfo(deviceId, pDeviceInfo);
    }

    virtual synStatus getDynamicShapesTensorInfoArrayV2(synStreamHandle             streamHandle,
                                                        synRecipeHandle             recipeHandle,
                                                        std::vector<tensor_info_t>& tensorInfoArray) const
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getDynamicShapesTensorInfoArrayV2(streamHandle, recipeHandle, tensorInfoArray);
    }

    virtual synStatus getModuleId(uint32_t& idOut)
    {
        VERIFY_ORIGINAL_IMPL_RET_UNSUPPORTED
        return m_originalImpl->getModuleId(idOut);
    }
};

#endif /*_SYN_SYN_SINGLETON_INTERFACE_H_*/
