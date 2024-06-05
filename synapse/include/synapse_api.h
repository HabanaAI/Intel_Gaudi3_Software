/***************************************************************************
 * Copyright (C) 2019 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 ****************************************************************************
 */

#ifndef SYNAPSE_API_H
#define SYNAPSE_API_H

#include <stdint.h>
#include <stdbool.h>
#include "synapse_api_types.h"
#include "synapse_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SYN_API_CALL     __stdcall

#else
#define SYN_API_CALL
#include <linux/types.h>
#endif

// Stream related functions

//!
/*!
 ***************************************************************************************************
 *   @brief Create a stream.
 *
 *   @param pStreamHandle     [out] Returned handle of newly created stream
 *   @param deviceId          [in]  Device ID connected to stream
 *   @param flags             [in]  Parameters for stream creation
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStreamCreateGeneric(synStreamHandle*  pStreamHandle,
                                              const synDeviceId deviceId,
                                              const uint32_t    flags);

//!
/*!
 ***************************************************************************************************
 *   @brief Destroy a stream
 *
 *   Destroys the stream specified by streamHandle.
 *
 *   In case the device is still doing work in the stream streamHandle when synStreamDestroy() is called,
 *   the function will return immediately with error synBusy.
 *   This function is not thread safe (if one thread is destroying a stream while other thread is
 *   using the stream, the behavior is not defined.
 *
 *   @param streamHandle      [in]  Stream to be destroyed
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStreamDestroy( const synStreamHandle streamHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Makes a stream wait on an event.
 *
 *   Makes all future work submitted to streamHandle wait for completion of eventHandle. eventHandle needs to be
 *   registered to other stream than streamHandle. It works on the same device only (not between devices)
 *
 *   @param streamHandle      [in]  Stream to wait
 *   @param eventHandle       [in]  Event to wait on
 *   @param flags             [in]  Parameters for the operation (must be 0)
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStreamWaitEvent( const synStreamHandle       streamHandle,
                                           synEventHandle              eventHandle,
                                           const uint32_t              flags );

//!
/*!
 ***************************************************************************************************
 *   @brief Waits for all commands in stream to complete.
 *
 *   Blocking function; Waits until the device has completed all operations in the stream specified
 *   by streamHandle.
 *   This function is not thread-safe by design - other threads are not blocked on the stream unless
 *   they call this function as well.
 *
 *   @param streamHandle      [in]  Stream to wait for
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStreamSynchronize( const synStreamHandle streamHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Determine status of a compute stream.
 *
 *   Returns synSuccess if stream idle or synBusy if there are pending operations in the stream
 *
 *   @param streamHandle      [in]  Stream to check
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStreamQuery( const synStreamHandle streamHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Create an event
 *
 *   Creates an event *pEventHandle for the given device
 *
 *   @param pEventHandler     [out] Returned handle of newly created event
 *   @param deviceId          [in]  Device to associate newly created event with
 *   @param flags             [in]  Flags for the operation
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventCreate( synEventHandle*       pEventHandler,
                                       const synDeviceId     deviceId,
                                       const uint32_t        flags );

//!
/*!
 ***************************************************************************************************
 *   @brief Destroy an event
 *
 *   Destroys the event specified by eventHandle.
 *   An event may be destroyed before it is complete. In this case, the call does not block on
 *   completion of the event, and any associated resources will automatically be released
 *   asynchronously at completion.
 *
 *   @param eventHandle       [in]  Event to be destroyed
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventDestroy( synEventHandle eventHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Records an event.
 *
 *  Captures in eventHandle the contents of streamHandle at the time of this call. eventHandle and streamHandle must be
 *  from the same device. Calls such as synEventQuery() or synStreamWaitEvent() will then examine
 *  or wait for completion of the work that was captured. Uses of streamHandle after this call do not
 *  modify eventHandle.
 *  synEventRecord() can be called multiple times on the same event and will overwrite the
 *  previously captured state. Other APIs such as synStreamWaitEvent() use the most recently
 *  captured state at the time of the API call, and are not affected by later calls to
 *  synEventRecord(). Before the first call to synEventRecord(), an event represents an empty set of
 *  work, so for example synEventQuery() would return synSuccess.
 *  recording to the same handle from two different threads is not thread safe
 *
 *   @param eventHandle       [in]  Event to record
 *   @param streamHandle      [in]  Stream to record event for
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventRecord( synEventHandle          eventHandle,
                                       const synStreamHandle   streamHandle );

/*!
 ***************************************************************************************************
 *   @brief Queries an event's status.
 *
 *  Queries the status of work currently captured by eventHandle. Returns synSuccess if all captured work
 *  has been completed, or synBusy if any captured work is incomplete.
 *
 *   @param eventHandle       [in]  Event to check
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventQuery( const synEventHandle eventHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Waits for an event to complete.
 *
 *  Blocking function; Waits until the completion of all work currently captured in eventHandle
 *
 *   @param eventHandle       [in]  Event to wait for
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventSynchronize( const synEventHandle eventHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Computes the elapsed time between two events.
 *
 *  Computes the elapsed time between two events. Or if eventHandleEnd is 0, returns the timestamp of eventHandleStart.
 *  If synEventRecord() has not been called on either event then synObjectNotInitialized is returned.
 *  If synEventRecord() has been called on both events but one or both of them has not yet been completed
 *  (that is, synEventQuery() would return synBusy on at least one of the events), synBusy is returned.
 *   @param pNanoSeconds              [out] number of elapsed nanoseconds, or nanosecond timestamp of start event
 *   @param eventHandleStart          [in]  Start event
 *   @param eventHandleEnd            [in]  Stop event
 *
 *   @return                  The status of the operation
 *                            synInvalidArgument: One or both events do not collect time
 *                            synUnavailable:     The time of one or both events is already gone (too old)
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventElapsedTime( uint64_t*               pNanoSeconds,
                                            const synEventHandle    eventHandleStart,
                                            const synEventHandle    eventHandleEnd );

//!
/*!
 ***************************************************************************************************
 * @brief Binds synEventHandle(s) to tensor(s)-id.
 *
 * Mark eventHandle(s) in relation to external tensor(s). EventHandle and launchTensorsInfo must be
 * from the same device. This API is used after recipe compilation, before calling synLaunch.
 * synEventMapTensor() can be called multiple times on the same event and will overwrite the
 * previously captured state. Before the first call to synEventMapTensor(), an event represents an empty set of
 * work, so for example synEventQuery() would return synSuccess.
 * Mapping to the same handle from two different threads is not thread safe.
 *
 * @param eventHandles      [in]      Event(s) to record
 * @param numOfEvents       [in]      Number of events
 * @param launchTensorsInfo [in]      List of external tensors to bind
 * @param recipeHandle      [in]      Recipe handle of current tensors
 *
 * @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventMapTensor(synEventHandle*                eventHandles,
                                         size_t                         numOfEvents,
                                         const synLaunchTensorInfo*     launchTensorsInfo,
                                         const synRecipeHandle          recipeHandle);

//!
/*!
 ***************************************************************************************************
 * @brief Binds synEventHandle(s) to tensor(s)-id.
 *
 * Mark eventHandle(s) in relation to external tensor(s). EventHandle and launchTensorsInfoExt must be
 * from the same device. This API is used after recipe compilation, before calling synLaunch.
 * synEventMapTensorExt() can be called multiple times on the same event and will overwrite the
 * previously captured state. Before the first call to synEventMapTensorExt(), an event represents an empty set of
 * work, so for example synEventQuery() would return synSuccess.
 * Mapping to the same handle from two different threads is not thread safe.
 *
 * @param eventHandles         [in]      Event(s) to record
 * @param numOfEvents          [in]      Number of events
 * @param launchTensorsInfoExt [in]      List of external tensors to bind
 * @param recipeHandle         [in]      Recipe handle of current tensors
 *
 * @return                     The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synEventMapTensorExt(synEventHandle*                eventHandles,
                                            size_t                         numOfEvents,
                                            const synLaunchTensorInfoExt*  launchTensorsInfoExt,
                                            const synRecipeHandle          recipeHandle);

//!
/*!
 ***************************************************************************************************
 * @brief   Extracts execution order of external tensors from a recipe.
 *
 * @param   recipeHandle         [in] The Synapse recipe to extract from
 * @param   numOfExternalTensors [in] Number of external tensors queried
 * @param   tensorIds            [out] TensorIds out value to be extracted
 *                                     (to be allocated by caller at least numOfExternalTensors)
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorExtExtractExecutionOrder(const      synRecipeHandle recipeHandle,
                                                         uint32_t   numOfExternalTensors,
                                                         uint64_t*  tensorIds);

//!
/*!
 ***************************************************************************************************
 * @brief   Destroys the recipe handle stored in host memory
 *
 * @param   recipeHandle         [in] The Synapse recipe to destroy
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synRecipeDestroy (synRecipeHandle recipeHandle );

//!
/*!
 ***************************************************************************************************
 * @brief   serialize a recipe to disk
 *
 * @param   recipeHandle             [in]  The Recipe to serialize
 * @param   recipeFileName           [in]  The filename to serialize to
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synRecipeSerialize( const synRecipeHandle  recipeHandle,
                                           const char*            recipeFileName );

/*!
 ***************************************************************************************************
 * @brief   Deserialize a recipe from disk
 *
 * @param   pRecipeHandle            [out] A pointer to the Recipe to deserialize
 * @param   recipeFileName           [in]  The filename to read from
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synRecipeDeSerialize (synRecipeHandle*  pRecipeHandle,
                                             const char*       recipeFileName );


//!
/*!
 ***************************************************************************************************
 * @brief   Extract information from recipe (via recipe handle, located in host memory)
 *
 * @param   retVal          [out] Returned array of value of requested attributes
 * @param   recipeAttr      [in]  Array of attributes to query of type synRecipeAttribute
 * @param   querySize       [in]  Size of array of attributes to query (and of the retVal array)
 * @param   recipeHandle    [in]  The handle of the inquired recipe.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */

synStatus SYN_API_CALL synRecipeGetAttribute( uint64_t*                 retVal,
                                              const synRecipeAttribute* recipeAttr,
                                              const unsigned            querySize,
                                              const synRecipeHandle     recipeHandle);

//!
/*!
***************************************************************************************************
* @brief Create a memory section.
*
* @param sectionHandle         [out] Returned handle of newly created section
* @param sectionDescriptor     [in]  Deprecated - should not be used
* @param graph                 [in]  The Synapse graph in which a node is created
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionCreate( synSectionHandle*        sectionHandle,
                                         uint64_t                 sectionDescriptor,
                                         const synGraphHandle     graph );

//!
/*!
***************************************************************************************************
* @brief Sets the group of a memory section.
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionGroup          [in]  Group of the section (0-255)
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionSetGroup( synSectionHandle         sectionHandle,
                                           uint64_t                 sectionGroup);

//!
/*!
***************************************************************************************************
* @brief Get the group of a memory section.
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionGroup          [out] A pointer to the group
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionGetGroup( synSectionHandle         sectionHandle,
                                           uint64_t*                sectionGroup);

//!
/*!
***************************************************************************************************
* @brief Sets whether a memory section is managed by the application (persistent).
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionIsPersistent   [in]  Persistency value
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionSetPersistent( synSectionHandle   sectionHandle,
                                                bool               sectionIsPersistent);

//!
/*!
***************************************************************************************************
* @brief Gets whether a memory section is managed by the application (persistent).
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionIsPersistent   [out] A pointer to persistency value
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionGetPersistent( synSectionHandle     sectionHandle,
                                                bool*                sectionIsPersistent);

//!
/*!
***************************************************************************************************
* @brief Sets whether a memory section is const.
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionIsConst        [in]  Consistency value
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionSetConst( synSectionHandle   sectionHandle,
                                           bool               sectionIsConst);

//!
/*!
***************************************************************************************************
* @brief Gets whether a memory section is const.
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionIsConst        [out] A pointer to consistency value
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionGetConst( synSectionHandle     sectionHandle,
                                           bool*                sectionIsConst);

//!
/*!
***************************************************************************************************
* @brief Gets A pointer to the property data of this sectionId.
*
* Right now this API supports only const section
* The section ID can be found per tensor by calling synTensorRetrieveLaunchInfoById
* and using synRetrievedLaunchTensorInfo::tensorSectionId
*
* @param pRecipeHandle         [in]  Handle of a created recipe
* @param synSectionId          [in]  ID of the required section
* @param prop                  [in]  Section property type
* @param propertyPtr           [out] A pointer to the requested property of this sectionId
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synRecipeSectionGetProp(const synRecipeHandle  pRecipeHandle,
                                               const synSectionId     sectionId,
                                               const synSectionProp   prop,
                                               uint64_t*              propertyPtr);

//!
/*!

***************************************************************************************************
* @brief Sets whether a memory section should have read-modify-write (RMW) capability.
*
* Note that the maximal capacity of such a section and all other such sections that are
* simultaneously used is limited. The limit can be read using synDeviceGetAttribute by
* reading the DEVICE_ATTRIBUTE_MAX_RMW_SIZE attribute.
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionIsRMW          [in]  RMW value
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionSetRMW( synSectionHandle   sectionHandle,
                                         bool               sectionIsRMW);

//!
/*!
***************************************************************************************************
* @brief Gets whether a memory section has read-modify-write (RMW) capability.
*
* @param sectionHandle         [in]  Handle of a created section
* @param sectionIsRMW          [out] A pointer to RMW value
*
* @return                      The status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synSectionGetRMW( synSectionHandle     sectionHandle,
                                         bool*                sectionIsRMW);

//!
/*!
 ***************************************************************************************************
 * @brief Clears the requested const-sections' host-buffer from given recipe, if exists.
 *
 * This API will clear the host-buffer(s) from the given recipe (relevant for const-sections only).
 * It will not be unmaped and will not be deallocated.
 *
 * @param recipeHandle      [in] A handle to the recipe which manages the host-buffer(s) requested to be clear.
 * @param sectionIds        [in] A pointer to an array of section-ids defining which const-sections holds the
 *                               host-buffers requested to be clear.
 * @param numOfSections     [in] The amount of sections which their host-buffers requested to be clear.
 *
 * @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synRecipeSectionHostBuffersClear( synRecipeHandle     recipeHandle,
                                                         const synSectionId* sectionIds,
                                                         size_t              numOfSections );

//!
/*!
 ***************************************************************************************************
 * @brief   *DEPRECATED* Launches a recipe on the specified device stream.
 *
 * This API will invoke the graph represented by the recipe handle,
 * on the stream HW resources. The recipe inputs and outputs are denoted
 * by the synLaunchTensorInfo array holding the current invocation tensor
 * details like size and address.
 * The tensor ascription will be done via the tensor name for each tensor
 * in the array.
 *
 * @param   streamHandle              [in]  Stream to enqueue operation to
 * @param   launchTensorsInfo         [in]  A pointer to a list of structs holding the tensor
 *                                          information
 * @param   numberOfTensors           [in]  The number of tensors in launchTensorsInfo
 * @param   pWorkspace                [in]  A pointer to workspace buffer
 * @param   pRecipeHandle             [in]  The RecipeHandle holding the recipe to execute
 * @param   flags                     [in]  A bit map indicates one or more of the following values:
 *                                          SYN_FLAGS_TENSOR_NAME: identify the tensors by their names,
 *                                          instead of their ids.
 *
 * @return                            Status of the operation
 ***************************************************************************************************
 */

synStatus SYN_API_CALL synLaunch( const synStreamHandle             streamHandle,
                                  const synLaunchTensorInfo*        launchTensorsInfo,
                                  const uint32_t                    numberOfTensors,
                                  uint64_t                          pWorkspace,
                                  const synRecipeHandle             pRecipeHandle,
                                  uint32_t                          flags);

//!
/*!
 ***************************************************************************************************
 * @brief   *DEPRECATED* Launches a recipe on the specified device stream.
 *
 * This API will invoke the graph represented by the recipe handle,
 * on the stream HW resources. The recipe inputs and outputs are denoted
 * by the synLaunchTensorInfoExt array holding the current invocation tensor
 * details like size and address.
 * The tensor ascription will be done via the tensor name for each tensor
 * in the array.
 *
 * @param   streamHandle                 [in]  Stream to enqueue operation to
 * @param   launchTensorsInfoExt         [in]  A pointer to a list of structs holding the tensor
 *                                             information
 * @param   numberOfTensors              [in]  The number of tensors in launchTensorsInfo
 * @param   pWorkspace                   [in]  A pointer to workspace buffer
 * @param   pRecipeHandle                [in]  The RecipeHandle holding the recipe to execute
 * @param   flags                        [in]  A bit map indicates one or more of the following values:
 *                                             SYN_FLAGS_TENSOR_NAME: identify the tensors by their names,
 *                                             instead of their ids.
 *
 * @return                               Status of the operation
 ***************************************************************************************************
 */

synStatus SYN_API_CALL synLaunchExt(const synStreamHandle             streamHandle,
                                    const synLaunchTensorInfoExt*     launchTensorsInfoExt,
                                    const uint32_t                    numberOfTensors,
                                    uint64_t                          pWorkspace,
                                    const synRecipeHandle             pRecipeHandle,
                                    uint32_t                          flags);

//!
/*!
 ***************************************************************************************************
 * @brief Launches a recipe on the specified device stream.
 *
 * This API will invoke the graph represented by the recipe handle,
 * on the stream HW resources. The recipe inputs and outputs are denoted
 * by the synLaunchTensorInfoExt array holding the current invocation tensor
 * details like size and address.
 * The tensor ascription will be done via the tensor ID for each tensor
 * in the array, unless specified by the flags parameter to do it via name.
 * The user can query the tensor ids before hand with the synTensorRetrieveIds
 * API.
 * This API also receives external events, which are events mapped to external tensors.
 * In order to synchronize with them, when the suitable tensor's outputs are ready for given events,
 * the API synStreamWaitEvent can be used.
 *
 * @param   streamHandle              [in]  Stream to enqueue operation to
 * @param   launchTensorsInfoExt      [in]  A pointer to a list of structs holding the tensor
 *                                          information
 * @param   numberOfTensors           [in]  The number of tensors in launchTensorsInfo
 * @param   pWorkspace                [in]  A pointer to workspace buffer
 * @param   pRecipeHandle             [in]  The RecipeHandle holding the recipe to execute
 * @param   eventHandleList           [in]  The event handles related to external tensors to be submitted
 * @param   numberOfEvents            [in]  The number of external events
 * @param   flags                     [in]  A bit map indicates one or more of the following values:
 *                                          SYN_FLAGS_TENSOR_NAME: identify the tensors by their names,
 *                                          instead of their ids.
 *
 * @return                            Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synLaunchWithExternalEvents(const synStreamHandle          streamHandle,
                                                   const synLaunchTensorInfo*     launchTensorsInfoExt,
                                                   const uint32_t                 numberOfTensors,
                                                   uint64_t                       pWorkspace,
                                                   const synRecipeHandle          pRecipeHandle,
                                                   synEventHandle*                eventHandleList,
                                                   const uint32_t                 numberOfEvents,
                                                   uint32_t                       flags);

//!
/*!
 ***************************************************************************************************
 * @brief Launches a recipe on the specified device stream.
 *
 * This API will invoke the graph represented by the recipe handle,
 * on the stream HW resources. The recipe inputs and outputs are denoted
 * by the synLaunchTensorInfoExt array holding the current invocation tensor
 * details like size and address.
 * The tensor ascription will be done via the tensor ID for each tensor
 * in the array, unless specified by the flags parameter to do it via name.
 * The user can query the tensor ids before hand with the synTensorRetrieveIds
 * API.
 * This API also receives external events, which are events mapped to external tensors.
 * In order to synchronize with them, when the suitable tensor's outputs are ready for given events,
 * the API synStreamWaitEvent can be used.
 *
 * @param   streamHandle              [in]  Stream to enqueue operation to
 * @param   launchTensorsInfoExt      [in]  A pointer to a list of structs holding the tensor
 *                                          information
 * @param   numberOfTensors           [in]  The number of tensors in launchTensorsInfo
 * @param   pWorkspace                [in]  A pointer to workspace buffer
 * @param   pRecipeHandle             [in]  The RecipeHandle holding the recipe to execute
 * @param   eventHandleList           [in]  The event handles related to external tensors to be submitted
 * @param   numberOfEvents            [in]  The number of external events
 * @param   flags                     [in]  A bit map indicates one or more of the following values:
 *                                          SYN_FLAGS_TENSOR_NAME: identify the tensors by their names,
 *                                          instead of their ids.
 *
 * @return                            Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synLaunchWithExternalEventsExt(const synStreamHandle          streamHandle,
                                                      const synLaunchTensorInfoExt*  launchTensorsInfoExt,
                                                      const uint32_t                 numberOfTensors,
                                                      uint64_t                       pWorkspace,
                                                      const synRecipeHandle          pRecipeHandle,
                                                      synEventHandle*                eventHandleList,
                                                      const uint32_t                 numberOfEvents,
                                                      uint32_t                       flags);

//!
/*!
 ***************************************************************************************************
 *   @brief Destroy a memory section.
 *
 *   @param sectionHandle     [in] a handle to the section to destroy
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synSectionDestroy( synSectionHandle sectionHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Gets the size of the workspace which is required to execute a given recipe
 *
 *   @param pWorkspaceSize    [out] the size of the workspace in bytes
 *   @param recipeHandle      [in] a handle to the recipe that is queried
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synWorkspaceGetSize( uint64_t*                pWorkspaceSize,
                                            const synRecipeHandle    recipeHandle );

//!
/*!
 ***************************************************************************************************
 * @brief   Multi-entry memory copy between the device and host asynchronously
 *
 * @param   streamHandle   [in]  Stream to enqueue operation to
 * @param   src            [in]  Pointer to an array of source addresses to read from
 * @param   size           [in]  Pointer to an array of amounts in bytes to read
 * @param   dst            [in]  Pointer to an array of dst addresses to write to
 * @param   direction      [in]  The direction to memcpy
 * @param   numCopies      [in]  Amount of elements passed in the above [in] arrays
 *
 *
 * @return                 Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synMemCopyAsyncMultiple(const synStreamHandle      streamHandle,
                                               const uint64_t*            src,
                                               const uint64_t*            size,
                                               const uint64_t*            dst,
                                               const synDmaDir            direction,
                                               const uint64_t             numCopies);

//!
/*!
 ***************************************************************************************************
 * @brief   Memory copy between the device and host asynchronously
 *
 * @param   streamHandle   [in]  Stream to enqueue operation to
 * @param   src            [in]  The source address to read from
 * @param   size           [in]  The amount in bytes to read
 * @param   dst            [in]  The dst address to write to
 * @param   direction      [in]  The direction to memcpy
 *
 * @return                 Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synMemCopyAsync(  const synStreamHandle     streamHandle,
                                         const uint64_t            src,
                                         const uint64_t            size,
                                         const uint64_t            dst,
                                         const synDmaDir           direction );

//!
/*!
 ***************************************************************************************************
 *   @brief Return number of compute-capable devices.
 *
 *   @param pCount       [out] Returned number of devices
 *
 *   @return            The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetCount( uint32_t* pCount );

//!
/*!
 ***************************************************************************************************
 *   @brief Return number of compute-capable devices ( by device type ).
 *
 *   @param pCount       [out] Returned number of devices
 *   @param deviceType   [in]  Device type to count
 *
 *   @return             The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetCountByDeviceType( uint32_t*              pCount,
                                                      const synDeviceType    deviceType );

//!
/*!
***************************************************************************************************
 *   @brief Return number of compute-capable devices by device type.
 *
 *   @param count       [out] Returned number of devices for each device type
 *
 *   @return            The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceCount( uint32_t count[synDeviceTypeSize] );

//!
/*!
***************************************************************************************************
 *   @brief Return an array of compute-capable devices module ID's.
 *
 *   Each HPU has an ID (0-7) configured and burnt to the HW. These module ID's can be used to differentiate between
 *   different HPU's at the same server. This API will return an array of the module ID's the host can work with.
 *   @param   pDeviceModuleIds  [out] Array that holds the different ID's.
 *   @param   size              [in/out]  The array size user allocated (Can we queried with synDeviceGetCount)
 *                                        Upon success, this param will hold the number of Id's/
 *
 * *
 *   @return            The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetModuleIDs( uint32_t *pDeviceModuleIds, uint32_t*  size);

//!
/*!
 ***************************************************************************************************
 *   @brief Acquire a device (by device-type)
 *
 *   @param pDeviceId           [out] The acquired device-id.
 *   @param deviceType          [in]  The device-type requested to be acquired.
 *                                    In cfase of an invalid-type, finds the first device
 *                                    regardless of its type
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceAcquireByDeviceType( synDeviceId*            pDeviceId,
                                                     const synDeviceType     deviceType);

//!
/*!
 ***************************************************************************************************
 *   @brief Acquire a device (by Module-Id)
 *
 *   @param pDeviceId           [out] The acquired device-id.
 *   @param moduleId            [in]  The Module-Id the requested device is associated with.
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceAcquireByModuleId( synDeviceId*      pDeviceId,
                                                   const synModuleId moduleId );

//!
/*!
 ***************************************************************************************************
 *   @brief Acquire a device (by PCI-bus)
 *
 *   @param pDeviceId           [out] The acquired device-id.
 *   @param pciBus              [in]  The PCI-Bus the requested device resides on.
 *                                    In case of an empty string (or nullptr), finds the first
 *                                    device found
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceAcquire( synDeviceId*   pDeviceId,
                                         const char*    pciBus );

//!
/*!
 ***************************************************************************************************
 *   @brief Wait for compute device to finish
 *
 *   Blocks until the device has completed all preceding requested tasks
 *   Returns an error if one of the preceding tasks has failed
 *
 *   @param deviceId          [in]  Device requested to be synchronized
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceSynchronize( const synDeviceId     deviceId );

//!
/*!
 ***************************************************************************************************
 * @brief   Returns the approximate Synapse driver version.
 *
 * Returns an ASCII string identifying Synapse driver version in convention x.y.z.w
 * Len specifies the maximum length of the string that may be returned.
 * @param   pDriverVersion   [out] driver version
 * @param   len              [in]  Maximum length of string to store in name
 *
 * @return                   Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDriverGetVersion( char*        pDriverVersion,
                                            const int    len );

//!
/*!
 ***************************************************************************************************
 * @brief   Returns an identifier string for the device.
 *
 * Updates an ASCII string identifying the device deviceId in the NULL-terminated string pointed to
 * by pName (buffer allocated by the user).
 * The len specifies the maximum length of the string that may be returned.
 *
 * @param   pName     [out] Returned identifier string for the device
 * @param   len       [in]  Maximum length of string to store in name
 * @param   deviceId  [in]  The ID of the requested device
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetName( char*               pName,
                                         const int           len,
                                         const synDeviceId   deviceId );

//!
/*!
 ***************************************************************************************************
 * @brief   Returns an identifier string for the device PCI bus-ID.
 *
 * Updates an ASCII NULL-terminated string, pointed to by pPciBusId (buffer allocated by the user),
 * containing the PCI bus-ID of a given device (deviceId).
 * The len parameter specifies the maximum length of the string that may be returned.
 *
 * @param   pName     [out] Returned identifier string for the device' PCI bus-ID
 * @param   len       [in]  Maximum length of the string to be stored at pPciBusId
 * @param   deviceId  [in]  The ID of the requested device
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetPCIBusId( char*               pPciBusId,
                                             const int           len,
                                             const synDeviceId   deviceId );

//!
/*!
 ***************************************************************************************************
 * @brief   *DEPRECATED* Creates a tensor object
 *
 * Created tensors are only tensor handles (name + shape & type
 * definitions) for graph compilation and do not contain any data or
 * references to any buffers.
 *
 * This API is deprecated. Please use synTensorHandleCreate instead to create an
 * empty tensor handle and other tensor APIs to set properties for this tensor
 * according to your requirements:
 *
 *  -  synTensorAssignToSection
 *  -  synTensorSetHostPtr
 *  -  synTensorSetGeometry
 *  -  synTensorSetDeviceFullLayout
 *
 * @param   pTensor             [out]  The tensor that was created
 * @param   descriptor          [in]   A previously-created tensor descriptor
 * @param   pSectionHandle      [in]   Section handle where the tensor resides at
 * @param   sectionDescriptor   [in]   The offset in bytes from the given section base address
 *
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorCreate( synTensor*                        pTensor,
                                        const synTensorDescriptor*        descriptor,
                                        const synSectionHandle            pSectionHandle,
                                        const uint64_t                    sectionOffset);

//!
/*!
 ***************************************************************************************************
 * @brief   *DEPRECATED* Creates a constant tensor object
 *
 * Created constant tensors are inputs for graph compilation and internally
 * keep a pointer to user allocated data. The user can free the data buffer
 * after compilation.
 *
 * This API is deprecated. Please use synTensorHandleCreate instead to create an
 * empty tensor handle and synTensorSetHostPtr to mark it as a const tensor and
 * add the host buffer. Use other APIs to add more properties to the tensor,
 * such as synTensorSetGeometry and synTensorSetDeviceFullLayout.
 *
 * @param   pTensor         [out]  The constant tensor that was created.
 * @param   descriptor      [in]   A previously-created tensor descriptor.
 *
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synConstTensorCreate (synTensor*                        pTensor,
                                             const synTensorDescriptor*        descriptor );

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves launch tensors amount
 *
 * Retrieve the amount of tensors that should be given to the synLaunch APIs.
 *
 * @param   pRecipeHandle   [in]     The inquired recipe
 * @param   numOfTensors    [out]    Number of tensors in recipe
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveLaunchAmount(const synRecipeHandle   pRecipeHandle,
                                                     uint32_t*               numOfTensors);

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves launch tensors IDs
 *
 * Retrieve the IDs of tensors that should be given to the synLaunch APIs.
 * The number of tensors should be queried using synTensorRetrieveLaunchAmount
 * before calling this API.
 *
 * @param   pRecipeHandle   [in]    The inquired recipe
 * @param   tensorsIds      [out]   An array of IDs
 * @param   numOfTensors    [in]    Number of tensors' IDs to get
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveLaunchIds(const synRecipeHandle pRecipeHandle,
                                                  uint64_t*             tensorsIds,
                                                  const uint32_t        numOfTensors);

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves launch tensors information
 *
 * Allows the user to query launch tensors in a given recipe.
 * Fills out a synRetrievedLaunchTensorInfo struct containing information about the launch tensor for
 * each queried tensor.
 * The number of launch tensors and their IDs should be queried using
 * synTensorRetrieveLaunchAmount and synTensorRetrieveLaunchIds before calling this API.
 *
 * @param   pRecipeHandle       [in]     The inquired recipe
 * @param   numOfTensors        [in]     Number of tensors to get their infos
 * @param   tensorsLaunchInfo   [in/out] A pointer to the synRetrievedLaunchTensorInfo array of
 *                                       size numOfTensors.
 *                                       Each given synRetrievedLaunchTensorInfo
 *                                       must contain the ID of a tensor to be queried.
 *                                       The rest of the synRetrievedLaunchTensorInfo struct will
 *                                       be filled by API.
 *                                       For any synRetrievedLaunchTensorInfo which will have an
 *                                       invalid ID, its tensorType will be set into
 *                                       TENSOR_TYPE_INVALID.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveLaunchInfoById(const synRecipeHandle         pRecipeHandle,
                                                       const uint32_t                numOfTensors,
                                                       synRetrievedLaunchTensorInfo* tensorsLaunchInfo );

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves launch tensors information
 *
 * Allows the user to query launch tensors in a given recipe.
 * Fills out a synRetrievedLaunchTensorInfoExt struct containing information about the launch tensor for
 * each queried tensor.
 * The number of launch tensors and their IDs should be queried using
 * synTensorRetrieveLaunchAmount and synTensorRetrieveLaunchIds before calling this API.
 *
 * @param   pRecipeHandle          [in]     The inquired recipe
 * @param   numOfTensors           [in]     Number of tensors to get their infos
 * @param   tensorsLaunchInfoExt   [in/out] A pointer to the synRetrievedLaunchTensorInfoExt array of
 *                                          size numOfTensors.
 *                                          Each given synRetrievedLaunchTensorInfoExt
 *                                          must contain the ID of a tensor to be queried.
 *                                          The rest of the synRetrievedLaunchTensorInfoExt struct will
 *                                          be filled by API.
 *                                          For any synRetrievedLaunchTensorInfoExt which will have an
 *                                          invalid ID, its tensorType will be set into
 *                                          TENSOR_TYPE_INVALID.
 *
 * @return                         Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveLaunchInfoByIdExt(const synRecipeHandle            pRecipeHandle,
                                                          const uint32_t                   numOfTensors,
                                                          synRetrievedLaunchTensorInfoExt* tensorsLaunchInfoExt );

//!
/*!
***************************************************************************************************
* @brief   Retrieves information for a launch tensor that has multiple info structs
*
* @param   pRecipeHandle       [in]     The inquired recipe
* @param   tensorId            [in]     Id of the tensor to get infos for
* @param   tensorLaunchInfo    [in/out] A pointer to the synRetrievedLaunchTensorInfo array of
*                                       size numLaunchInfos. If given as nullptr, then the API
*                                       will return the number of launch infos to allocate in
*                                       the numLaunchInfos param.
* @param   numLaunchInfos      [in/out] number of launch infos (either filled by synapse or given
*                                       by user in case the tensorLaunchInfo is allocated)
*
* @return                  Status of the operation
***************************************************************************************************
*/
synStatus SYN_API_CALL synTensorRetrieveMultiLaunchInfosById(const synRecipeHandle         pRecipeHandle,
                                                             const uint64_t                tensorId,
                                                             synRetrievedLaunchTensorInfo* tensorLaunchInfo,
                                                             uint32_t*                     numLaunchInfos);

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves persistent tensors' information by name
 *
 * Allows the user to query specific persistent tensors in a given recipe.
 * Fills out a TensorMetaData struct containing information about the persistent tensor
 * for each queried tensor.
 *
 * @param   pRecipeHandle        [in]     The inquired recipe
 * @param   numOfTensors         [in]     Number of tensors to get their infos
 * @param   tensorsMetadataInfo  [in/out] A pointer to the TensorMetadataInfo array of size numOfTensors.
 *                                        Each given TensorMetadataInfo must contain the name of a
 *                                        tensor to be queried. the rest of the TensorMetadataInfo
 *                                        struct will be filled by API.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveInfosByName(const synRecipeHandle   pRecipeHandle,
                                                    const uint32_t          numOfTensors,
                                                    TensorMetadataInfo*     tensorsMetadataInfo );

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves persistent tensors' information by name
 *
 * Allows the user to query specific persistent tensors in a given recipe.
 * Fills out a TensorMetaData struct containing information about the persistent tensor
 * for each queried tensor.
 *
 * @param   pRecipeHandle           [in]     The inquired recipe
 * @param   numOfTensors            [in]     Number of tensors to get their infos
 * @param   tensorsMetadataInfoExt  [in/out] A pointer to the TensorMetadataInfoExt array of size numOfTensors.
 *                                           Each given TensorMetadataInfoExt must contain the name of a
 *                                           tensor to be queried. the rest of the TensorMetadataInfoExt
 *                                           struct will be filled by API.
 *
 * @return                                   Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveInfosByNameExt(const synRecipeHandle      pRecipeHandle,
                                                       const uint32_t             numOfTensors,
                                                       TensorMetadataInfoExt*     tensorsMetadataInfoExt );

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves Persistent Tensors' amount
 *
 * @param   pRecipeHandle        [in]     The inquired recipe
 * @param   numOfTensors         [out]    Number of tensors in recipe
  *
 * @return                  Status of the operation
 ***************************************************************************************************
 */

synStatus SYN_API_CALL synTensorRetrievePersistentAmount(const synRecipeHandle   pRecipeHandle,
                                                         uint32_t*               numOfTensors);

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves Persistent Tensors' names
 *
 * @param   pRecipeHandle        [in]    The inquired recipe
 * @param   tensorsName          [out]   An array of strings
 * @param   numOfTensors         [in]    Number of tensors' names to get
  *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveNames(const synRecipeHandle   pRecipeHandle,
                                              char                    tensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                                              const uint32_t          numOfTensors);


//!
/*!
 ***************************************************************************************************
 * @brief   Retrieves Tensors id by name
 *
 * @param   pRecipeHandle       [in]  The inquired recipe
 * @param   tensorNames         [in]  A pointer to an array of tensor names
 * @param   tensorIds           [out] An array, allocated by caller, of tensor ids.
 *                                    filled by synapse.
 * @param   numOfTensors        [in]  Number of tensors in each array.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorRetrieveIds(const synRecipeHandle    pRecipeHandle,
                                            const char**             tensorNames,
                                            uint64_t*                tensorIds,
                                            const uint32_t           numOfTensors);

//!
/*!
 ***************************************************************************************************
 * @brief   Destroy a tensor object
 *
 * Inside Synapse tensor objects are ref counted, so if this function is called after assigning
 * the tensor to some graph node, the tensor is valid (till synGraphDestroy).
 *
 * Upon calling synGraphDestroy, all tensors connected to the graph will also be destroyed, so
 * calling synTensorDestroy on these tensors afterwards is invalid.
 *
 * @param   tensor          [in]  The tensor to destroy
 *
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorDestroy( const synTensor tensor );

//!
/*!
 ***************************************************************************************************
 * @brief   Creates a node as part of the specified graph
 *
 *
 * @param   graphHandle          [in] The Synapse graph in which a node is created
 * @param   pInputsTensorList    [in] An array of input tensors
 * @param   pOutputsTensorList   [in] An array of output tensors
 * @param   numberInputs         [in] The amount of inputs
 * @param   numberOutputs        [in] The amount of outputs
 * @param   pUserParams          [in] a pointer to a user-defined struct containing the scalar arguments
 *                               to the kernel, that will be forwarded as-is to the glue code (see
 *                               the appropriate spec). It can be null.
 * @param   paramsSize           [in] The size in bytes of paramsSize
 * @param   pGuid                [in] the identifier of the operator. SynapseAI attempts to match it
 *                               against the list of pre-defined operators, and if no match is
 *                               found, it forwards it to the glue code library that reported
 *                               supporting this GUID in its GetKernelNames entry point. GUID
 *                               length is limited up to 30 characters.
 * @param   pName                [in] A user-defined name that will be later used in logging and graph
 *                               visualization. It’s recommended but not mandatory for this to be
 *                               unique.
 * @param   inputLayouts         [in] An array of strings which size pertains to the number of inputs.
 *                               Every entry in these arrays is the expected data layout for this
 *                               operand (e.g. “NHWC”) or null signifying the operator is
 *                               agnostic to the layout.
 * @param   outputLayouts        [in] As above, for the outputs.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeCreate( const synGraphHandle    graphHandle,
                                      const synTensor*        pInputsTensorList,
                                      const synTensor*        pOutputsTensorList,
                                      const uint32_t          numberInputs,
                                      const uint32_t          numberOutputs,
                                      const void*             pUserParams,
                                      const unsigned          paramsSize,
                                      const char*             pGuid,
                                      const char*             pName,
                                      const char**            inputLayouts,
                                      const char**            outputLayouts );

//!
/*!
 ***************************************************************************************************
 * @brief   Creates a node as part of the default graph with an option to get a unique ID
 *
 *
 * @param   graphHandle          [in] The Synapse graph in which a node is created
 * @param   pInputsTensorList    [in] An array of input tensors
 * @param   pOutputsTensorList   [in] An array of output tensors
 * @param   numberInputs         [in] The amount of inputs
 * @param   numberOutputs        [in] The amount of outputs
 * @param   pUserParams          [in] a pointer to a user-defined struct containing the scalar arguments
 *                               to the kernel, that will be forwarded as-is to the glue code (see
 *                               the appropriate spec). It can be null.
 * @param   pGuid                [in] the identifier of the operator. SynapseAI attempts to match it
 *                               against the list of pre-defined operators, and if no match is
 *                               found, it forwards it to the glue code library that reported
 *                               supporting this GUID in its GetKernelNames entry point. GUID
 *                               length is limited up to 30 characters.
 * @param   pName                [in] A user-defined name that will be later used in logging and graph
 *                               visualization. It’s recommended but not mandatory for this to be
 *                               unique.
 * @param   nodeUniqueId         [out] The API will return a unique ID for the new node,
 *                               that can be used for later references to this node by other API calls.
 * @param   inputLayouts         [in] An array of strings which size pertains to the number of inputs.
 *                               Every entry in these arrays is the expected data layout for this
 *                               operand (e.g. “NHWC”) or null signifying the operator is
 *                               agnostic to the layout.
 * @param   outputLayouts        [in] As above, for the outputs.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeCreateWithId( const synGraphHandle graphHandle,
                                            const synTensor*     pInputsTensorList,
                                            const synTensor*     pOutputsTensorList,
                                            const uint32_t       numberInputs,
                                            const uint32_t       numberOutputs,
                                            const void*          pUserParams,
                                            const unsigned       paramsSize,
                                            const char*          pGuid,
                                            const char*          pName,
                                            synNodeId*           nodeUniqueId,
                                            const char**         inputLayouts,
                                            const char**         outputLayouts );

//!
/*!
 ***************************************************************************************************
 * @brief   Set node type precision
 *
 *
 * @param   graphHandle         [in] The Synapse graph in which the node is created.
 * @param   guid                [in] A string identifier for the node type.
 * @param   precision           [in] The precision to set for all nodes for the given guid.
 *                                   If the given precision is syn_type_na, then it will be treated as
 *                                   "don't care", meaning it will take its type from its surrounding nodes.
 *
 * @return                      Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeTypeSetPrecision( const synGraphHandle graphHandle,
                                                const char*          guid,
                                                synDataType          precision );

//!
/*!
 ***************************************************************************************************
 * @brief   Creates a control dependency between two group of nodes
 *
 *
 * @param   graphHandle          [in] The Synapse graph in which the dependency is created
 * @param   pBlockingNodesIdList [in] An array of blocking node unique IDs
 * @param   pBlockedNodesIdList  [in] An array of blocked node unique IDs
 * @param   numberInputs         [in] The amount of blocking nodes
 * @param   numberOutputs        [in] The amount of blocked nodes
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeDependencySet( const synGraphHandle   graphHandle,
                                             const synNodeId*       pBlockingNodesIdList,
                                             const synNodeId*       pBlockedNodesIdList,
                                             const uint32_t         numberblocking,
                                             const uint32_t         numberblocked );

//!
/*!
 ***************************************************************************************************
 * @brief   Controls whether to use a deterministic implementation for a node (default is false)
 *
 *
 * @param   graphHandle          [in] The Synapse graph in which the node was created
 * @param   nodeId               [in] node unique id, as received from synNodeCreateWithId
 * @param   useDeterministic     [in] value to set; if true Node will use deterministic implementation
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeSetDeterministic( const synGraphHandle   graphHandle,
                                                const synNodeId        nodeId,
                                                const bool             useDeterministic);

//!
/*!
 ***************************************************************************************************
 * @brief   Gets Node deterministic implementation state
 *
 *
 * @param   graphHandle          [in] The Synapse graph in which the node was created
 * @param   nodeId               [in] node unique id, as received from synNodeCreateWithId
 * @param   useDeterministic     [out] pointer to where to fill the data
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeGetDeterministic( const synGraphHandle   graphHandle,
                                                const synNodeId        nodeId,
                                                bool*                  useDeterministic);

//!
/*!
 ***************************************************************************************************
 * @brief   Configure the following rounding mode per MME node (default is Round-Nearest-Even)
 *
 *
 * @param   graphHandle          [in] The Synapse graph in which the node was created
 * @param   nodeId               [in] node unique id, as received from synNodeCreateWithId
 * @param   RoundingMode         [in] one of the rounding mode options in synRoundingMode to set for the node
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeSetRoundingMode(  const synGraphHandle    graphHandle,
                                                const synNodeId         nodeId,
                                                const synRoundingMode   roundingMode);

//!
/*!
 ***************************************************************************************************
 * @brief   Gets Node RoundingMode implementation state per MME node
 *
 *
 * @param   graphHandle          [in] The Synapse graph in which the node was created
 * @param   nodeId               [in] node unique id, as received from synNodeCreateWithId
 * @param   RoundingMode         [out] pointer to return the node's rounding mode
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeGetRoundingMode(  const synGraphHandle    graphHandle,
                                                const synNodeId         nodeId,
                                                synRoundingMode*        pRoundingMode);
//!
/*!
 ***************************************************************************************************
 * @brief   Compile the graph specified
 *
 * @param   pRecipeHandle       [out] Handle to a HabanaRecipe
 * @param   graphHandle         [in] The Synapse graph to compile
 * @param   pRecipeName         [in] The name of the recipe that will be generated
 * @param   pBuildLog           [in] A compilation output log file name. Can be Null
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphCompile( synRecipeHandle*                pRecipeHandle,
                                        const synGraphHandle            graphHandle,
                                        const char*                     pRecipeName,
                                        const char*                     pBuildLog );

//!
/*!
 ***************************************************************************************************
 * @brief   Create a new empty graph instance for the given device type.
 *
 * @param   pGraphHandle        [out] The created Synapse graph
 * @param   deviceType          [in]  The device type the graph is created for
 *
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphCreate( synGraphHandle*        pGraphHandle,
                                       const synDeviceType    deviceType );

//!
/*!
 *  ***************************************************************************************************
 * @brief   *DEPRECATED* Set attributes for a given graph to be used during the compilation process
 *
 * This API is deprecated. Please use synGraphSetAttributes instead.
 *
 * @param   graphHandle     [in] Synapse graph handle
 * @param   attributes      [in] Array of attributes to set of type synGraphAttribute
 * @param   values          [in] Array of values corresponding to each attribute in the previous array
 * @param   size            [in] Size of the attributes/values arrays
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphSetAttribute(synGraphHandle           graphHandle,
                                            const synGraphAttribute* attributes,
                                            const uint64_t*          values,
                                            const uint32_t           size);

//!
/*!
 *  ***************************************************************************************************
 * @brief   Set attributes for a given graph to be used during the compilation process
 *
 * @param   graphHandle     [in] Synapse graph handle
 * @param   attributes      [in] Array of attributes to set of type synGraphAttribute
 * @param   values          [in] Array of values corresponding to each attribute in the previous array
 * @param   size            [in] Size of the attributes/values arrays
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphSetAttributes(synGraphHandle              graphHandle,
                                             const synGraphAttribute*    attributes,
                                             const synGraphAttributeVal* values,
                                             const uint32_t              size);

//!
/*!
 *  ***************************************************************************************************
 * @brief   *DEPRECATED* Get attributes for a given graph that will be used during the compilation process
 *
 * This API is deprecated. Please use synGraphGetAttributes instead.
 *
 * @param   graphHandle     [in]  Synapse graph handle
 * @param   attributes      [in]  Array of attributes to get of type synGraphAttribute
 * @param   values          [out] Array of values corresponding to each attribute in the previous array
 * @param   size            [in]  Size of the attributes/values arrays
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphGetAttribute(synGraphHandle           graphHandle,
                                            const synGraphAttribute* attributes,
                                            uint64_t*                values,
                                            const uint32_t           size);

//!
/*!
 *  ***************************************************************************************************
 * @brief   Get attributes for a given graph that will be used during the compilation process
 *
 * @param   graphHandle     [in]  Synapse graph handle
 * @param   attributes      [in]  Array of attributes to get of type synGraphAttribute
 * @param   values          [out] Array of values corresponding to each attribute in the previous array
 * @param   size            [in]  Size of the attributes/values arrays
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphGetAttributes(synGraphHandle           graphHandle,
                                             const synGraphAttribute* attributes,
                                             synGraphAttributeVal*    values,
                                             const uint32_t           size);

//!
/*!
 ***************************************************************************************************
 * @brief   Create a new empty graph instance for the given device type in Eager mode.
 *
 * @param   pGraphHandle        [out] The created Synapse graph
 * @param   deviceType          [in]  The device type the graph is created for
 *
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphCreateEager( synGraphHandle*        pGraphHandle,
                                            const synDeviceType    deviceType );

//!
/*!
 ***************************************************************************************************
 * @brief   Duplicates a given graph including all of its nodes and tensors.
 *
 * A mapping of nodes and tensors is returned to allow the user to make subsequent
 * manipulation to the returned graph (requiring handles to newly created objects).
 * In case either of the passed tensorsMap or nodesMap is nullptr, the API will
 * only fill the correct values for numTensors and numNodes.
 *
 * @param   graphHandle        [in]      The input Synapse graph
 * @param   newGraphHandle     [out]     The newly created Synapse graph
 * @param   tensorsMap         [out]     an array to hold the mapping of existing to duplicated tensors
 * @param   numTensors         [in/out]  size of user supplied tensorsMap array
                                         (filled by Synapse if passed tensorsMap is nullptr)
 * @param   nodesMap           [out]     an array to hold the mapping of existing to duplicated nodes
 * @param   numNodes           [in/out]  size of user supplied nodesMap array
 *                                       (filled by Synapse if passed nodesMap is nullptr)
 *
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphDuplicate( synGraphHandle         graphHandle,
                                          synGraphHandle*        newGraphHandle,
                                          synTensorHandleMap*    tensorsMap,
                                          uint32_t*              numTensors,
                                          synNodeHandleMap*      nodesMap,
                                          uint32_t*              numNodes );

//!
/*!
 ***************************************************************************************************
 * @brief   Infer Graph max shapes for Eager Graph.
 *
 * This API is used for graphs created by a previous call to synGraphDuplicate.
 * It will infer the max shapes of all intermediate output tensors without a previously set max shape.
 * For output tensor with a previously provided max shape, the API will validate it against the infered shape.
 * Failure to infer an output tensor shape or validation failure would result in synFail being returned.
 * Any persistent tensor with a missing max shape will result in validation failure and synFail being returned.
 * In case of success, all graph tensors would have their corresponding max shape set.
 *
 * @param   graphHandle     [in] The duplicated synGraphHandle
 *
 * @return                  synSuccess if all maxDims were successfully infered.
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphInferShapes( synGraphHandle    graphHandle );

//!
/*!
 ***************************************************************************************************
 * @brief   Destroys the current graph instance
 * @param   graphHandle    [in] The Synapse graph to destroy
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphDestroy( const synGraphHandle    graphHandle );

//!
/*!
 ***************************************************************************************************
 * @brief   Gets device type of a current graph instance
 * @param   graphHandle    [in]  The Synapse graph to query
 * @param   deviceType     [out] The given graph's device type
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synGraphGetDeviceType(const synGraphHandle    graphHandle,
                                             synDeviceType*          deviceType);

//!
/*!
 ***************************************************************************************************
 * @brief   Sets device memory
 *
 * @param   pDeviceMem      [in] Pointer to device memory
 * @param   value           [in] Value to set
 * @param   numOfElements   [in] Number of elements
 * @param   streamHandle    [in] Stream to enqueue operation to
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synMemsetD32Async( uint64_t                 pDeviceMem,
                                          const uint32_t           value,
                                          const size_t             numOfElements,
                                          const synStreamHandle    streamHandle );

//!
/*!
 ***************************************************************************************************
 * @brief   Sets device memory
 *
 * @param   pDeviceMem      [in] Pointer to device memory
 * @param   value           [in] Value to set
 * @param   numOfElements   [in] Number of elements
 * @param   streamHandle    [in] Stream identifier
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synMemsetD8Async( uint64_t                 pDeviceMem,
                                         const unsigned char      value,
                                         const size_t             numOfElements,
                                         const synStreamHandle    streamHandle );

//!
/*!
 ***************************************************************************************************
 * @brief   Sets device memory4
 *
 * @param   pDeviceMem      [in] Pointer to device memory
 * @param   value           [in] Value to set
 * @param   numOfElements               [in] Number of elements
 * @param   streamHandle    [in] Stream identifier
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synMemsetD16Async( uint64_t                 pDeviceMem,
                                          const uint16_t           value,
                                          const size_t             numOfElements,
                                          const synStreamHandle    streamHandle );

//!
/*!
 ***************************************************************************************************
 *   @brief Creates a memory allocation on the host and maps it in the device's MMU
 *
 *   @param deviceId        [in]  The device id for resource creation.
 *   @param size            [in]  Size of the created buffer in bytes.
 *   @param flags           [in]  flags for the operation. should be zero
 *   @param buffer          [out] A pointer to the newly created buffer.
 *
 *   @return                The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synHostMalloc( const synDeviceId      deviceId,
                                      const uint64_t         size,
                                      const uint32_t         flags,
                                      void**                 buffer );

/*!
 ***************************************************************************************************
 *   @brief Deletes a memory allocation on the host
 *
 *   @param deviceId        [in]  The device id for resource manipulation.
 *   @param buffer          [in]  A pointer to the buffer to be deleted
 *   @param flags           [in]  flags for the operation. should be zero
 *
 *   @return                The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synHostFree( const synDeviceId    deviceId,
                                    const void*          buffer,
                                    const uint32_t       flags );

//!
/*!
 ***************************************************************************************************
 *   @brief Maps a given buffer (allocated by the user on the host) to a memory-allocation of a
 *   specific device
 *
 *   @param deviceId        [in]  The device id for resource mapping.
 *   @param size            [in]  Size in bytes, of the buffer to be mapped.
 *   @param buffer          [in] A pointer to the buffer requested to be mapped.
 *
 *   @return                The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synHostMap( const synDeviceId          deviceId,
                                   const uint64_t             size,
                                   const void*                buffer );

//!
/*!
 ***************************************************************************************************
 *   @brief Un-map a host memory allocation
 *
 *   @param deviceId        [in]  The device id for resource creation.
 *   @param buffer          [in]  A pointer to the buffer to be unmapped
 *
 *   @return                The status of the operation.
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synHostUnmap( const synDeviceId    deviceId,
                                     const void*          buffer );

//!
/*!
 ***************************************************************************************************
 *   @brief Creates a memory allocation on a specific device
 *
 *   @param deviceId        [in]  The device id for resource creation.
 *   @param size            [in]  Size of the created buffer in bytes.
 *   @param reqAddr         [in]  The requested address of the buffer that is allocated. This request
 *                                  serves as a hint. Synapse is not required to provide the given
 *                                  address as the malloc result. User is required to check what is
 *                                  the actual address that synapse provided by inspecting the content
 *                                  of 'buffer' argument.
 *                                  Malloc will succeed regardless if Synapse can or can't provide the requested address.
 *                                  Setting reqAddr = 0 implies that the user is indifferent to the address
 *                                  provided.
 *                                  Its the user responsibility to ask for an 128 bytes aligned address.
 *   @param flags           [in]  flags for the operation. should be zero
 *   @param buffer          [out] A pointer to the newly created buffer.
 *
 *   @return                The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceMalloc( const synDeviceId    deviceId,
                                        const uint64_t       size,
                                        uint64_t             reqAddr,
                                        const uint32_t       flags,
                                        uint64_t*            buffer );

//!
/*!
 ***************************************************************************************************
 *   @brief Deletes a memory allocation on the device
 *
 *   @param deviceId        [in]  The device id for resource manipulation.
 *   @param buffer          [in]  A pointer to the buffer to be deleted
 *   @param flags           [in]  flags for the operation. should be zero
 *
 *   @return                The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceFree( const synDeviceId     deviceId,
                                      const uint64_t        buffer,
                                      const uint32_t        flags );

//!
/*!
 ***************************************************************************************************
 * @brief Initiate the synapse instance
 *
 * synInitialize must be called before any other synapse api except synDriverGetVersion.
 * After finishing using synapse api - call synDestroy.
 *
 * The second invocation of synInitialize does not make any affect and returns synSuccess.
 * if HABANA_DISABLE_DOUBLE_SYN_INITIALIZE environment variable is set
 * then the second invocation of synInitialize returns synAlreadyInitialized.
 *
 * @return                        The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synInitialize();

//!
/*!
 ***************************************************************************************************
 * @brief Destroy the synapse instance
 *
 * It is the responsibility of the caller to ensure that no synapse API are in use
 * while synDestroy() is executing.
 *
 * @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDestroy();

//!
/*!
 ***************************************************************************************************
 *   @brief Release a device (and destroy data allocated for it)
 *
 *   @param deviceId            [in] The device-id requested to be released.
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceRelease( const synDeviceId deviceId );

//!
/*!
 ***************************************************************************************************
 *   @brief Return the free and total memory on a specific device
 *
 *   @param deviceId    [in]  The device id the memory info is asked for
 *   @param free        [out] Free memory available
 *   @param total       [out] Total memory on device
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetMemoryInfo( const synDeviceId    deviceId,
                                               uint64_t*            free,
                                               uint64_t*            total );

//!
/*!
 ***************************************************************************************************
 * @brief   *DEPRECATED* Get the Synapse's Device-Info
 *
 * @param   deviceId        [in]  The inquired (user's) ID of the device
 * @param   pDeviceInfo     [out] The requested Device-Info
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetInfo( const synDeviceId      deviceId,
                                         synDeviceInfo*         pDeviceInfo );

//!
/*!
 ***************************************************************************************************
 * @brief   Get the Synapse's Device-Info
 *
 * @param   deviceId        [in]  The inquired (user's) ID of the device
 * @param   pDeviceInfo     [out] The requested Device-Info
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetInfoV2( const synDeviceId      deviceId,
                                           synDeviceInfoV2*       pDeviceInfo );

//!
/*!
 ***************************************************************************************************
 * @brief   Query the given device attributes
 *
 * @param   retVal          [out] Returned array of value of requested attributes
 * @param   deviceAttr      [in]  Array of attributes to query of type synDeviceAttribute
 * @param   querySize       [in]  Size of array of attributes to query (and of the retVal array)
 * @param   deviceId        [in]  The inquired (user's) ID of the device
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetAttribute( uint64_t*                 retVal,
                                              const synDeviceAttribute* deviceAttr,
                                              const unsigned            querySize,
                                              const synDeviceId         deviceId);

//!
/*!
 ***************************************************************************************************
 * @brief   Query the given device's attributes (identified by its module-ID)
 *
 * @param   retVal          [out] Returned array of value of requested attributes
 * @param   deviceAttr      [in]  Array of attributes to query of type synDeviceAttribute
 * @param   querySize       [in]  Size of array of attributes to query (and of the retVal array)
 * @param   moduleId        [in]  The Module-Id the requested device is associated with
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetAttributeByModuleId( uint64_t*                 retVal,
                                                        const synDeviceAttribute* deviceAttr,
                                                        const unsigned            querySize,
                                                        const synModuleId         moduleId );

/*!
 ***************************************************************************************************
 * @brief Get the Synapse's Device Attribute according to Device Type
 *
 * @param retVal     [out] Returned array of value of requested attributes
 * @param deviceAttr [in] Array of attributes to query of type synDeviceAttribute
 * @param querySize  [in] Size of array of attributes to query (and of the retVal array)
 * @param deviceType [in] The inquired device type
 *
 * @return Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceTypeGetAttribute( uint64_t*                 retVal,
                                                  const synDeviceAttribute* deviceAttr,
                                                  const unsigned            querySize,
                                                  const synDeviceType       deviceType);

//!
/*!
 ***************************************************************************************************
 * @brief   Set a Synapse configuration parameter
 *
 * @param   configurationName        [in]  The config parameter name
 * @param   configurationValue       [in]  The requested value to be set
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synConfigurationSet(const char*          configurationName,
                                           const char*          configurationValue);

//!
/*!
 ***************************************************************************************************
 * @brief   Get the value of a Synapse configuration parameter
 *
 * @param   configurationName        [in]  The config parameter name
 * @param   configurationValue      [out]  The requested value to be set
 * @param   size                     [in]  configurationValue buffer size
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synConfigurationGet(const char*          configurationName,
                                           char*                configurationValue,
                                           uint64_t             size);


//!
/*!
 ***************************************************************************************************
 * @brief   Query if synapse profiler requires HBM memory allocated from the user.
 *
 * If 0 is returned then synProfilerStart can be immediately called.
 * Otherwise the user needs to allocate a buffer with the size returned by "bytesRequired",
 * And provide it to the profiler with synProfilerSetUserBuffer, before synProfilerStart can be called.
 *
 * @param   deviceId                [in]   The inquired (user's) ID of the device
 * @param   bytesRequired           [out]  Amount of bytes required by synapse profiler to successfully
 *                                         start a profiling session.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerQueryRequiredMemory(const synDeviceId deviceId, uint32_t* bytesRequired);


//!
/*!
 ***************************************************************************************************
 * @brief   Supply synapse profiler with user allocated HBM memory
 *
 * Required size of the buffer should be supplied by synProfilerQueryRequiredMemory.
 * The memory will be used by synapse profiler for all consecutive profiling sessions.
 *
 * @param   deviceId                [in]   The inquired (user's) ID of the device
 * @param   userBuffer              [out]  User allocated buffer
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerSetUserBuffer(const synDeviceId deviceId, void* userBuffer);


//!
/*!
 ***************************************************************************************************
 * @brief   Begins profiling session using on-device instrumentation
 *
 * @param   type              [in]  The requested type of trace
 * @param   deviceId          [in]  The inquired (user's) ID of the device (set to 0 for host trace)
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerStart( const synTraceType        type,
                                         const synDeviceId         deviceId );

//!
/*!
 ***************************************************************************************************
 * @brief   Ends profiling session and saves output to file(s)
 *
 * @param   type              [in]  The requested type of trace
 * @param   deviceId          [in]  The inquired (user's) ID of the device (set to 0 for host trace)
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerStop( const synTraceType      type,
                                        const synDeviceId       deviceId );

//!
/*!
 ***************************************************************************************************
 * @brief   Retrieve the trace buffer from memory
 *
 * @param   type        [in] The requested type of trace to retrieve (device or host)
 * @param   deviceId    [in] The inquired (user's) ID of the device (ignored for host trace)
 * @param   format      [in] The requested format of the trace data
 * @param   buffer      [in/out] Pointer to user allocated memory to contain the requested buffer.
 *                            If null, only size and numEntries will be returned.
 *                            The returned buffer is built in the following format:
 *                            [synTraceEvent enries][strings pool][size_t num][size_t version]
 *                            num: Amount of synTraceEvent entries
 *                            version: Synprof parser version.
 *                            It's important to make sure that this buffer is not freed as long
 *                            as the synTraceEvents entries are still accessed, for not invalidating
 *                            the entries char pointers from the strings pool.
 * @param   size        [in/out] Input is the size of the allocated buffer in bytes, output is
 *                              the actual amount of trace data copied to buffer. If buffer is null,
 *                              the size of the entire trace buffer (including the chars array) is
 *                              returned. If null, the trace buffer will be written to the file
 *                              system in the requested format.
 * @param   numEntries  [out] Actual amount of synTraceEvent entries copied to buffer, or the total
 *                            amount of synTraceEvent entries in the trace buffer if buffer is null.
 *
 * @return              The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerGetTrace( const synTraceType      type,
                                            const synDeviceId       deviceId,
                                            const synTraceFormat    format,
                                            void*                   buffer,
                                            size_t*                 size,
                                            size_t*                 numEntries );

//!
/*!
 ***************************************************************************************************
 * @brief   Gets profiler current time in nano seconds
 *
 * @param   nanoTime       [out] pointer to receive the time (in nano seconds)
 *
 * @return                 Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerGetCurrentTimeNS(uint64_t* nanoTime);


//!
/*!
 ***************************************************************************************************
 * @brief   Adds a profiler custom measurement
 *
 * The start time is given in the nanoTime parameter, and the end time is the
 * function call time.
 *
 * @param   description        [in] measurement description string
 * @param   nanoTime           [in] measurement start time in nano seconds
 *
 * @return                     Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerAddCustomMeasurement( const char* description,
                                                        uint64_t    nanoTime );


//!
/*!
 ***************************************************************************************************
 * @brief   Adds a profiler custom measurement with args
 *
 * The start time is given in the nanoTime parameter, and the end time is the
 * function call time. args are function arguments that will be visible in the profiler output
 *
 * @param   description        [in] measurement description string
 * @param   nanoTime           [in] measurement start time in nano seconds
 * @param   args               [in] array of arguments (null-terminated strings) to be shown
 *                                  in the profiler output, first element should be the arg
 *                                  name, second the value, and so on. i.e. the args should
 *                                  come in pairs.
 * @param   argsSize           [in] size of args array. Note that it should be an even number
 *                                  due to the args coming in pairs.
 *
 *
 * @return                     Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerAddCustomMeasurementArgs( const char*  description,
                                                            uint64_t     nanoTime,
                                                            const char** args,
                                                            size_t       argsSize );


//!
/*!
 ***************************************************************************************************
 * @brief   Adds a profiler custom measurement with args of a specific thread
 *
 * The start time is given in the nanoTime parameter, and the end time is the
 * function call time. args are function arguments that will be visible in the profiler output
 *
 * @param   description        [in] measurement description string
 * @param   nanoTime           [in] measurement start time in nano seconds
 * @param   args               [in] array of arguments (null-terminated strings) to be shown
 *                                  in the profiler output, first element should be the arg
 *                                  name, second the value, and so on. i.e. the args should
 *                                  come in pairs.
 * @param   argsSize           [in] size of args array. Note that it should be an even number
 *                                  due to the args coming in pairs.
 * @param   threadName         [in] name of event's thread
 *
 *
 * @return                     Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerAddCustomMeasurementArgsAndThread( const char*  description,
                                                                     uint64_t     nanoTime,
                                                                     const char** args,
                                                                     size_t       argsSize,
                                                                     const char*  threadName );

//!
/*!
 ***************************************************************************************************
 * @brief   Adds to the profiler a custom user event
 *
 * The start time and end time should come from the user using synProfilerGetCurrentTimeNS,
 * Args are function arguments that will be visible in the profiler output
 *
 * @param   description        [in] measurement description string
 * @param   startTime          [in] measurement start time in nano seconds,
 *                                  output of synProfilerGetCurrentTimeNS
 * @param   endTime            [in] measurement end time in nano seconds,
 *                                  similar to startTime
 * @param   args               [in] array of arguments (null-terminated strings) to be shown
 *                                  in the profiler output, first element should be the arg
 *                                  name, second the value, and so on. i.e. the args should
 *                                  come in pairs.
 * @param   argsSize           [in] size of args array. Note that it should be an even number
 *                                  due to the args coming in pairs.
 *
 *
 * @return                     Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synProfilerAddCustomEvent( const char*  description,
                                                  uint64_t     startTime,
                                                  uint64_t     endTime,
                                                  const char** args,
                                                  size_t       argsSize );

//!
/*!
 ***************************************************************************************************
 *   @brief Create an empty tensor handle.
 *
 *   The tensor is coupled with a graph.
 *   The tensor name may be omitted, in which case a default naming of Tensor_XXX will be generated.
 *   The tensor name must not be duplicated in other tensors within the graph.
 *
 *   After creating the empty tensor handle please use other tensor APIs to set properties
 *   for this tensor, according to your requirements:
 *   -  synTensorAssignToSection
 *   -  synTensorSetHostPtr
 *   -  synTensorSetGeometry
 *   -  synTensorSetDeviceFullLayout
 *   -  synTensorSetQuantizationData
 *   -  synTensorSetExternal
 *
 *   @param tensor            [out] A pointer to the created tensor handle.
 *   @param graph             [in]  A previously-created graph handle in which to create the tensor.
 *   @param type              [in]  The tensor classification in the graph.
 *   @param tensorName        [in]  The tensor's name - may be omitted (null or empty string).
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorHandleCreate(synTensor*               tensor,
                                             synGraphHandle           graph,
                                             synTensorType            type,
                                             const char*              tensorName);

//!
/*!
 ***************************************************************************************************
 *   @brief Assign a tensor to memory section and offset.
 *
 *   This API assigns a tensor to a previously-created section.
 *   Tensors for which no section was assigned reside in the workspace.
 *   section was returned from a previous successful call to synSectionCreate.
 *
 *   Tensor with host pointer can be assigned to a section only if the section is const.
 *
 *   Tensor with host pointer and a tensor without host pointer should not be assigned
 *   to the same section.
 *
 *   Device address cannot be set for non-persistent sections. In this case a status
 *   "synUnsupported" will be returned.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param section           [in] A previously-created section handle.
 *   @param byteOffset        [in] An offset in which the tensor resides within the section.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorAssignToSection(synTensor           tensor,
                                                synSectionHandle    section,
                                                uint64_t            byteOffset);

//!
/*!
 ***************************************************************************************************
 *   @brief set tensor section offset.
 *
 *   This API changes the tensor offset in the section it was previously assigned to.
 *   Calling this API for a tensor which was not previously assigned to a section
 *   will result in failure.
 *
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param byteOffset        [in] An offset in which the tensor resides within the section.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetSectionOffset(synTensor           tensor,
                                                 uint64_t            byteOffset);

//!
/*!
 ***************************************************************************************************
 *   @brief Sets a host buffer to the tensor.
 *
 *   This API sets the tensor as const (containing static data, such as weights or an embedding
 *   table).
 *   Synapse will copy the data internally and the user may release it.
 *   The dataType must be equal to the device data type set in synTensorSetDeviceFullLayout.
 *
 *   In inference graphs, the dataType may be different from the device data type if it is FP32.
 *   In addition, for inference graphs, the contents of the host-side data will be interpreted
 *   according to the quantization information for this data type supplied by synTensorSetQuantizationData.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param hostPtr           [in] A pointer to the host buffer.
 *   @param size              [in] The buffer size in bytes.
 *   @param dataType          [in] The buffer data type.
 *   @param copyBuffer        [in] Copy buffer content to the tensor (deprecated - should always pass true).
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetHostPtr(synTensor            tensor,
                                           void*                hostPtr,
                                           uint64_t             size,
                                           synDataType          dataType,
                                           bool                 copyBuffer);

//!
/*!
 ***************************************************************************************************
 *   @brief Sets quantization parameters to tensor.
 *
 *   This API sets the quantization metadata specified by prop based on the data in propVal.
 *   No more than propSize bytes will be read from propVal.
 *   Legal values to prop: SYN_QUANT_DYNAMIC_RANGE, SYN_QUANT_METADATA, SYN_FP_QUANT_METADATA, SYN_QUANT_FLAGS
 *
 *   For SYN_QUANT_DYNAMIC_RANGE, propVal should point to a synQuantDynamicRange struct to define
 *   the dynamic range of the tensor data. From the dynamic range, the quantization information of
 *   the tensor will be automatically calculated by the graph compiler.
 *
 *   For SYN_QUANT_METADATA / SYN_FP_QUANT_METADATA, propVal should point to a
 *   synQuantMetadata / synFpQuantMetadata struct to define the quantization information of the tensor
 *   for a specific data type - instead of relying on the automatic calculation done in the GC from
 *   the dynamic range. This prop can be called multiple times - once for each data type that quantization
 *   info should be set for.
 *   synQuantMetadata contains a pointer to an array of synQuantZPScale (zero point and scale).
 *   The size of this array is noted in numZPScales.
 *   synFpQuantMetadata contains a pointer to an array of synFpQuantParam (scale and exp bias).
 *   The size of this array is noted in numFpQuantParams.
 *   If this num is set to 1, the values are used for all channels in the specified tensor for
 *   that data type. Otherwise, it must equal the number of the channels in the tensor and indicates
 *   per-channel quantization for tensors that support it.
 *
 *   For SYN_QUANT_FLAGS, propVal should point to a synQuantFlags struct to define some quantization
 *   attributes of the tensor, to affect GC's automatic calculation of the quantization, mainly for
 *   MME weight tensors:
 *   - enablePerChannelQuant - calculate the zp and scale per channel, in tensors that support it.
 *   - isSparsifiedWeights -   mark the weights as sparse, and force the zp to be 0.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param prop              [in] The quantization property indicator
 *   @param propVal           [in] A pointer to user-allocated struct that matches prop
 *   @param propSize          [in] Size in bytes of propVal struct..
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetQuantizationData(synTensor               tensor,
                                                    synQuantizationProperty prop,
                                                    void*                   propVal,
                                                    uint64_t                propSize);

//!
/*!
 ***************************************************************************************************
 *   @brief Marks tensor as external/not-external.
 *
 *   This API sets the tensor as external.
 *   External tensors can be used to signal from graph during its execution.
 *
 *   @param tensor            [in] A previously-created tensor handle
 *   @param isExternal        [in] A flag to set tensor as (or non) external
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */


synStatus SYN_API_CALL synTensorSetExternal(synTensor               tensor,
                                            bool                    isExternal);

//!
/*!
 ***************************************************************************************************
 *   @brief Sets shape and dimension to tensor.
 *
 *   Set geometry according to geometryType.
 *   Legal values to geometryType: synGeometryMinSizes, synGeometryMaxSizes, synGeometrySizes and,
 *                                 synGeometryDims.
 *   If only one of synGeometryMinSizes or synGeometryMaxSizes is specified, the other is assumed
 *   to be identical (the same as using synGeometrySizes).
 *   synGeometryDims can be used to pass in the rank of the tensor without setting the shape,
 *   asking the compiler to infer them, if possible. In this case, only the dims field of the
 *   synTensorGeometry struct will be used, and the sizes field will be ignored.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param geometry          [in] A pointer to the synTensorGeometry struct.
 *   @param geometryType      [in] Specify if Minimum or Maximum sizes.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetGeometry(synTensor                   tensor,
                                            const synTensorGeometry*    geometry,
                                            synGeometryType             geometryType);

//!
/*!
 ***************************************************************************************************
 *   @brief Sets shape and dimension to tensor.
 *
 *   Set geometry according to geometryType.
 *   Legal values to geometryType: synGeometryMinSizes, synGeometryMaxSizes, synGeometrySizes and,
 *                                 synGeometryDims.
 *   If only one of synGeometryMinSizes or synGeometryMaxSizes is specified, the other is assumed
 *   to be identical (the same as using synGeometrySizes).
 *   synGeometryDims can be used to pass in the rank of the tensor without setting the shape,
 *   asking the compiler to infer them, if possible. In this case, only the dims field of the
 *   synTensorGeometryExt struct will be used, and the sizes field will be ignored.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param geometry          [in] A pointer to the synTensorGeometryExt struct.
 *   @param geometryType      [in] Specify if Minimum or Maximum sizes.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetGeometryExt(synTensor                   tensor,
                                               const synTensorGeometryExt* geometry,
                                               synGeometryType             geometryType);

//!
/*!
 ***************************************************************************************************
 *   @brief Sets a memory permutation on a tensor.
 *
 *   Set a permutation on a tensor that marks that the given tensor data is stored transposed in
 *   the memory according to this permutation on its dimensions.
 *
 *   The permutation array should store the order of the dimensions after their memory transpose.
 *
 *   Note: to unset a previously set permutation on this tensor, you can pass dims=0 in the
 *   permutation struct.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param permutation       [in] A pointer to the synTensorPermutation struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetPermutation(synTensor                   tensor,
                                               const synTensorPermutation* permutation);

//!
/*!
 ***************************************************************************************************
 *   @brief Sets device data type for a tensor.
 *
 *   Set the desired data type of the tensor in the device memory.
 *   This API replaces synTensorSetDeviceLayout/synTensorSetDeviceFullLayout which was used to set
 *   the data type along with strides (which are deprecated).
 *
 *   In inference graphs, a buffer provided by synTensorSetHostPtr will be quantized to the device
 *   data type, if is different from the buffer data type given in that API.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param deviceDataType    [in] The data type for the tensor on the device.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetDeviceDataType(synTensor   tensor,
                                                  synDataType deviceDataType);


//!
/*!
 ***************************************************************************************************
 *   @brief Sets device data type and strides.
 *
 *   Set the desired data type of the tensor and its strides in the device memory.
 *
 *   In inference graphs, a buffer provided by synTensorSetHostPtr will be quantized to the device
 *   data type, if is different from the buffer data type given in that API.
 *
 *   Currently, non-default strides are unsupported so the strides array should be set to zeros.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param layout            [in] A pointer to the synTensorFullDeviceLayout struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetDeviceFullLayout(synTensor                        tensor,
                                                    const synTensorDeviceFullLayout* layout);


//!
/*!
 ***************************************************************************************************
 *   @brief *DEPRECATED* - Sets device data type and strides.
 *
 *   Set the desired data type of the tensor and its strides in the device memory.
 *
 *   In inference graphs, a buffer provided by synTensorSetHostPtr will be quantized to the device
 *   data type, if is different from the buffer data type given in that API.
 *
 *   Currently, non-default strides are unsupported so the strides array should be set to zeros.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param layout            [in] A pointer to the synTensorDeviceLayout struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetDeviceLayout(synTensor                    tensor,
                                                const synTensorDeviceLayout* layout);

//!
/*!
 ***************************************************************************************************
 *   @brief Mark tensor as "allow permutation" on GC compilation
 *
 *   Marking a tensor as allowed permutation will let the graph compiler avoid transposing its data
 *   and keep the tensor permuted in the memory.
 *
 *   Tensor which is marked as allow permutation will not necessarily be permuted.
 *
 *   Enabled only for persistent tensors
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param allowpermutaion   [in] 1 / 0 -> allow or disallow (tensor is created with disallow as default).
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorSetAllowPermutation(synTensor tensor,
                                                    int8_t    allowPermutation);

//!
/*!
 ***************************************************************************************************
 *   @brief Get "allow permutation" attribute from tensor
 *
 *   See allow permutation definition on synTensorSetAllowPermutation description.
 *
 *   @param tensor            [in]  A previously-created tensor handle.
 *   @param allowPermutation  [out] 1 / 0 -> allow or disallow
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetAllowPermutation(synTensor tensor,
                                                    int8_t*   allowPermutation);

//!
/*!
 ***************************************************************************************************
 *   @brief Query a tensor's memory section.
 *
 *   This API returns the tensor's section handle and offset.
 *   For tensors without a section, null and 0 will be returned in section and byteOffset, respectively.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param section           [out]  The section handle allocated by the user
 *   @param byteOffset        [out]  The offset within the section
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetSection(synTensor           tensor,
                                           synSectionHandle*   section,
                                           uint64_t*           byteOffset);

//!
/*!
 ***************************************************************************************************
 *   @brief Query a tensor's host buffer.
 *
 *   This API returns the tensor's host buffer's pointer, size and data type.
 *   The tensor host buffer contains the tensor's static data.
 *   The data pointed to by hostPtr must not be released until the tensor is destroyed.
 *   Modifying the data pointed by hostPtr will modify the actual tensor data,
 *   and shouldn't be done after compiling the graph.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param hostPtr           [out]  The host buffer pointer
 *   @param size              [out]  The buffer size in bytes
 *   @param dataType          [out]  The buffer data type
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetHostPtr(synTensor            tensor,
                                           void**               hostPtr,
                                           uint64_t*            size,
                                           synDataType*         dataType);

//!
/*!
 ***************************************************************************************************
 *   @brief Query a tensor's quantization parameters.
 *
 *   This API returns the tensor's quantization properties.
 *   The quantization property type is determined by the prop argument.
 *   Legal values to prop: SYN_QUANT_DYNAMIC_RANGE, SYN_QUANT_METADATA, SYN_FP_QUANT_METADATA, SYN_QUANT_FLAGS
 *   The tensor's quantization property will be stored in a user-allocated struct pointed by propVal argument.
 *
 *   For SYN_QUANT_DYNAMIC_RANGE, propVal should point to a synQuantDynamicRange struct,
 *   which will be filled by the API.
 *
 *   For SYN_QUANT_METADATA / SYN_FP_QUANT_METADATA,
 *   propVal should point to a synQuantMetadata / synFpQuantMetadata struct with a specific data type.
 *   The struct contains an array of synQuantZPScale / synFpQuantParam structs which should be allocated by the user.
 *   In case the array is null, the returned struct will contain the channels number.
 *   In case the array isn't null, it's assumed allocated, and the returned struct will contain
 *   also the synQuantZPScale / synFpQuantParam array.
 *   The required allocation size is calculated as follows : number of tensor channels * sizeof(synQuantZPScale or synFpQuantParam)
 *
 *   For SYN_QUANT_FLAGS, propVal should point to a synQuantFlags struct which will be filled by the API.
 *
 *   The size of the property struct should be passed for validation. For SYN_QUANT_METADATA / SYN_FP_QUANT_METADATA,
 *   In case the synQuantZPScale / synQuantParam array is null, the size of synQuantMetadata / synFpQuantMetadata should be passed.
 *   In case the array is allocated, the size of the allocated array plus the size of
 *   synQuantMetadata / synFpQuantMetadata should be passed.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param prop              [in]   The quantization property indicator
 *   @param propVal           [out]  A pointer to user-allocated struct that matches prop
 *   @param propSize          [in]   Size in bytes of propVal struct
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetQuantizationData(synTensor               tensor,
                                                    synQuantizationProperty prop,
                                                    void*                   propVal,
                                                    uint64_t                propSize);

//!
/*!
 ***************************************************************************************************
 *   @brief Query tensor shape and dimension.
 *
 *   Geometry property will be returned in user-allocated buffer.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param geometry          [out]  A pointer to the synTensorGeometry struct
 *   @param geometryType      [in]   Type of the geometry to be queried
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetGeometry(const synTensor         tensor,
                                            synTensorGeometry*      geometry,
                                            synGeometryType         geometryType);

//!
/*!
 ***************************************************************************************************
 *   @brief Query tensor shape and dimension.
 *
 *   Geometry property will be returned in user-allocated buffer.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param geometry          [out]  A pointer to the synTensorGeometryExt struct
 *   @param geometryType      [in]   Type of the geometry to be queried
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetGeometryExt(const synTensor         tensor,
                                               synTensorGeometryExt*   geometry,
                                               synGeometryType         geometryType);

//!
/*!
 ***************************************************************************************************
 *   @brief Query if tensor is external.
 *
 *   Is external will be returned in user-allocated boolean
 *
 *   @param tensor            [in] A previously-created tensor handle
 *   @param isExternal        [out] A pointer to boolean
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */


synStatus SYN_API_CALL synTensorGetExternal(const synTensor         tensor,
                                            bool*                   isExternal);

//!
/*!
 ***************************************************************************************************
 *   @brief Gets the memory permutation of a tensor.
 *
 *   Returns the user-set permutation on the given tensor.
 *
 *   @param tensor            [in]  A previously-created tensor handle.
 *   @param permutation       [out] A pointer to the synTensorPermutation struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetPermutation(const synTensor       tensor,
                                               synTensorPermutation* permutation);

//!
/*!
 ***************************************************************************************************
 *   @brief Query the device data type for a tensor.
 *
 *   Get the desired data type of the tensor in the device memory.
 *
 *   @param tensor            [in] A previously-created tensor handle.
 *   @param deviceDataType    [out] The data type for the tensor on the device.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetDeviceDataType(synTensor    tensor,
                                                  synDataType* deviceDataType);

//!
/*!
 ***************************************************************************************************
 *   @brief *DEPRECATED* Query tensor device data type and strides.
 *
 *   Device layout property will be returned in user-allocated buffer.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param layout            [out]  A pointer to the synTensorDeviceLayout struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetDeviceLayout(const synTensor         tensor,
                                                synTensorDeviceLayout*  layout);

//!
/*!
 ***************************************************************************************************
 *   @brief Query tensor device data type and strides.
 *
 *   Device layout property will be returned in user-allocated buffer.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param layout            [out]  A pointer to the synTensorDeviceFullLayout struct.
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetDeviceFullLayout(const synTensor            tensor,
                                                    synTensorDeviceFullLayout* layout);

//!
/*!
 ***************************************************************************************************
 *   @brief Query tensor name.
 *
 *   Name will be returned in user-allocated buffer.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param size              [in]   Size in bytes of user allocated buffer
 *   @param name              [out]  Pointer to user-allocated buffer
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetName(const synTensor   tensor,
                                        const uint64_t    size,
                                        char*             name);

//!
/*!
 ***************************************************************************************************
 *   @brief Query tensor type.
 *
 *   Type (one of the values of the synTensorType enum) will be returned in user-allocated buffer.
 *
 *   @param tensor            [in]   A previously-created tensor handle
 *   @param type              [out]  Pointer to user-allocated buffer
 *
 *   @return                  The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTensorGetType(const synTensor   tensor,
                                        synTensorType*    type);

//!
/*!
 ***************************************************************************************************
 * @brief   Get the TPC libraries versions array size
 *
 * @param   size [out]  The number of TPC libreries
 *
 * @return  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTPCLibraryGetVersionSize(uint32_t* size);

//!
/*!
 ***************************************************************************************************
 * @brief   Get the TPC libraries paths and versions
 *          Each version in versions correlates to the library path in libs
 *
 * @param   libs     [out]  char** to hold an array of TPC libraries paths (char* to internally allocated map)
 *                          User should allocate char* * size taken from synTPCLibraryGetVersionSize API
 * @param   versions [out]  A pointer to an array of TPC libraries versions
 *                          Allocated by the user on the host with size taken from synTPCLibraryGetVersionSize API
 *
 * @return  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synTPCLibraryGetVersions(const char** libs, uint32_t* versions);

//!
/*!
 ***************************************************************************************************
 *   @brief Set stream affinity.
 *
 *   Stream affinity is set by round-robin manner by default when a stream is created.
 *   To change the default behavior use this API.
 *   Mask value indicates which HW queues are valid to choose.
 *   For example: 0x1 - only HW-Queue 0 is available, 0x2 - only HW-Queue 1 is available,
 *                0x3 - both HW-Queue 0 and HW-Queue 1 are available etc.
 *   For querying the possible affinity mask of the device, use synDeviceGetAffinityMaskRange
 *
 *   @param deviceId            [in]  Device ID connected to stream.
 *   @param pStreamHandle       [in]  Stream handle to change the affinity.
 *   @param streamAffinityMask  [in]  Stream affinity mask value.
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStreamSetAffinity(const synDeviceId       deviceId,
                                            const synStreamHandle   pStreamHandle,
                                            uint64_t                streamAffinityMask);

//!
/*!
 ***************************************************************************************************
 *   @brief Get stream affinity.
 *
 *   Mask value indicates which HW queue is in use for the provided stream handle.
 *   For example: 0x1 - HW-Queue 0 is in use, 0x2 - HW-Queue 1 is in use etc.
 *
 *   @param deviceId            [in]   Device ID connected to stream.
 *   @param pStreamHandle       [in]   Stream handle to query the affinity.
 *   @param streamAffinityMask  [out]  Stream affinity mask value.
 *
 *   @return                    The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStreamGetAffinity(const synDeviceId       deviceId,
                                            const synStreamHandle   pStreamHandle,
                                            uint64_t*               streamAffinityMask);

//!
/*!
 ***************************************************************************************************
 *   @brief Get device affinity possible range bit-mask.
 *
 *   A bit which is set indicates a HW-queue which is available.
 *   For example: 0x1 - only HW-Queue 0 is available, 0x2 - only HW-Queue 1 is available,
 *                0x3 - both HW-Queue 0 and HW-Queue 1 are available etc.
 *
 *   @param deviceId                    [in]  Device ID to query.
 *   @param deviceAffinityMaskRange     [out] Device affinity mask value.
 *
 *   @return                            The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetAffinityMaskRange(const synDeviceId  deviceId,
                                                     uint64_t*          deviceAffinityMaskRange);

//!
/*!
 ***************************************************************************************************
 *   @brief Get next device affinity number
 *
 *   The affinity number returned is based on round robin with respect to the device max affinity hardware queues.
 *   Different threads can use this function to sync between them to create streams with different
 *   streamAffinityMask.
 *
 *   @param deviceId                    [in]  Device ID to query.
 *   @param nextDeviceAffinity          [out] Device affinity number.
 *
 *   @return                            The status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synDeviceGetNextStreamAffinity(const synDeviceId deviceId, uint64_t* nextDeviceAffinity);

//!
/*!
 ***************************************************************************************************
 * @brief   Sets user params in a node
 *
 * This can be used to replace the user params given in the node creation.
 *
 * @param   graphHandle    [in] The Synapse graph in which the node was created
 * @param   nodeId         [in] node unique id, as received from synNodeCreateWithId
 * @param   userParams     [in] a pointer to a user-allocated struct containing the scalar arguments
 *                              to the node (definitions of the relevant structs for each node can be
 *                              found in the synapse and TPC spec docs).
 * @param   paramsSize     [in] size in bytes of the user params struct/buffer
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeSetUserParams( const synGraphHandle   graphHandle,
                                             const synNodeId        nodeId,
                                             const void*            userParams,
                                             const unsigned         paramsSize);

//!
/*!
 ***************************************************************************************************
 * @brief   Gets the user params of a node
 *
 * Copies the user params set in a node to a user-allocated buffer. The size of the buffer must match
 * the size of the struct set in the node.
 * To retrieve the size of the user params struct, the user can pass a null pointer for the user param,
 * in which case the paramsSize argument will be filled with the appropriate size.
 *
 * @param   graphHandle    [in] The Synapse graph in which the node was created
 * @param   nodeId         [in] node unique id, as received from synNodeCreateWithId
 * @param   userParams     [out] a pointer to a user-allocated buffer to copy the user params into.
 * @param   paramsSize     [in/out] a pointer to a variable containing the params size.
 *                                  in case the userParams buffer is allocated, this should be input as
 *                                  size in bytes of the userParams buffer - which must match the
 *                                  size of the struct in the node.
 *                                  if the userParams struct is null, then this param will be returned
 *                                  as output - the necessary size in bytes that the user
 *                                  should allocate.
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synNodeGetUserParams( const synGraphHandle   graphHandle,
                                             const synNodeId        nodeId,
                                             void*                  userParams,
                                             unsigned*              paramsSize);

//!
/*!
 ***************************************************************************************************
 * @brief Convert a synStatus's enum into string
 *
 * @param status            [in]  The synStatus's enum requested to be converted
 * @param statusDescription [out] The buffer which will hold the status-description
 * @param len               [in]  Maximum length of string to store in statusDescription
 *                                The maximum length of status-description is STATUS_DESCRIPTION_MAX_SIZE
 *
 * @return                  Status of the operation
 ***************************************************************************************************
 */
synStatus SYN_API_CALL synStatusGetBriefDescription(synStatus status, char* statusDescription, size_t len);

#ifdef __cplusplus
}

#endif

#endif //SYNAPSE_API_H
