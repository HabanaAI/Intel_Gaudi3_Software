/******************************************************************************
 * Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************/

/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2020 Intel Corp. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef HCCL_H_
#define HCCL_H_

// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <cstddef>              // for size_t
#include <cstdint>              // for uint64_t
#include "synapse_api_types.h"  // for synStreamHandle
#include "hccl_types.h"         // for hcclResult_t, hcclComm_t, hcclDataType_t

#define HCCL_P2P_SUPPORTED 1

#ifdef HCCL_WRAPPER_USE_STREAM
#if not HCCL_WRAPPER_USE_STREAM
#error "Current HCCL implementation requires stream usage."
#endif
#else
#define HCCL_WRAPPER_USE_STREAM 1
#endif

#define HCCL_MAJOR            2
#define HCCL_MINOR            6
#define HCCL_PATCH            4
#define HCCL_SUFFIX           ""
#define HCCL_VERSION_CODE     2604
#define HCCL_VERSION(X, Y, Z) ((X)*1000 + (Y)*100 + (Z))

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the HCCL_VERSION_CODE of the HCCL library.
 * This integer is coded with the MAJOR, MINOR and PATCH level of the HCCL library.
 */
hcclResult_t hcclGetVersion(int* version);

/* Generates an Id to be used in hcclCommInitRank. hcclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling hcclCommInitRank. */
hcclResult_t hcclGetUniqueId(hcclUniqueId* uniqueId);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a Habana device, which has to be set before calling
 * hcclCommInitRank.
 * hcclCommInitRank implicitly synchronizes with other ranks, so it must be
 * called by different threads/processes or use hcclGroupStart/hcclGroupEnd. */
hcclResult_t hcclCommInitRank(hcclComm_t* comm, int nranks, hcclUniqueId commId, int rank);

/* Creates a clique of communicators (single process version).
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(hcclComm_t).
 * If devlist is NULL, the first ndev Habana devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
hcclResult_t hcclCommInitAll(hcclComm_t* comm, int ndev, const int* devlist);

/* Waits for all submitted work of communicator to be done.
 * By doing so, prepares communicator for destruction. */
hcclResult_t hcclCommFinalize(hcclComm_t comm);

/* Frees resources associated with communicator object, but waits for any operations
 * that might still be running on the device. */
hcclResult_t hcclCommDestroy(hcclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
hcclResult_t hcclCommAbort(hcclComm_t comm);

/* Returns a human-readable error message. */
const char* hcclGetErrorString(hcclResult_t result);

/* Checks whether the comm has encountered any asynchronous errors */
hcclResult_t hcclCommGetAsyncError(hcclComm_t comm, hcclResult_t* asyncError);

/* Gets the number of ranks in the communicator clique. */
hcclResult_t hcclCommCount(hcclComm_t comm, int* count);

/* Returns the Habana device number associated with the communicator. */
hcclResult_t hcclCommSynDevice(hcclComm_t comm, int* device);

/* Returns the user-ordered "rank" associated with the communicator. */
hcclResult_t hcclCommUserRank(hcclComm_t comm, int* rank);

/* Returns FD for HBM memory region if it was registered for gaudi-direct. */
int hcclLookupDMABuff(uint64_t addr, uint64_t size, int* fd);

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the Habana stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */

/*
 * Reduce
 *
 * Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the Habana device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
hcclResult_t hcclReduce(const void*     sendbuff,
                        void*           recvbuff,
                        size_t          count,
                        hcclDataType_t  datatype,
                        hcclRedOp_t     reduceOp,
                        int             root,
                        hcclComm_t      comm,
                        synStreamHandle stream_handle);

/*
 * (deprecated) Broadcast (in-place)
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the Habana device) where data resides before the
 * operation is started.
 *
 * This operation is implicitely in place.
 */
hcclResult_t
hcclBcast(void* buff, size_t count, hcclDataType_t datatype, int root, hcclComm_t comm, synStreamHandle stream_handle);


/*
 * Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the Habana device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
hcclResult_t hcclBroadcast(const void*     sendbuff,
                           void*           recvbuff,
                           size_t          count,
                           hcclDataType_t  datatype,
                           int             root,
                           hcclComm_t      comm,
                           synStreamHandle stream_handle);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
hcclResult_t hcclAllReduce(const void*     sendbuff,
                           void*           recvbuff,
                           size_t          count,
                           hcclDataType_t  datatype,
                           hcclRedOp_t     reduceOp,
                           hcclComm_t      comm,
                           synStreamHandle stream_handle);

/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
hcclResult_t hcclReduceScatter(const void*     sendbuff,
                               void*           recvbuff,
                               size_t          recvcount,
                               hcclDataType_t  datatype,
                               hcclRedOp_t     reduceOp,
                               hcclComm_t      comm,
                               synStreamHandle stream_handle);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other HPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
hcclResult_t hcclAllGather(const void*     sendbuff,
                           void*           recvbuff,
                           size_t          sendcount,
                           hcclDataType_t  datatype,
                           hcclComm_t      comm,
                           synStreamHandle stream_handle);

/*
 * AlltoAll
 *
 * preform send and receive operations from all ranks\peers to all ranks\peers.
 *
 *
 */
hcclResult_t hcclAlltoAll(const void*     sendbuff,
                          void*           recvbuff,
                          size_t          count,
                          hcclDataType_t  datatype,
                          hcclComm_t      comm,
                          synStreamHandle stream_handle);

/*
 * Barrier
 * Wait on syncing between all the ranks in the communicator
 * This function is blocking only in TCP mode
 */
hcclResult_t hcclBarrier(hcclComm_t comm, synStreamHandle stream_handle);

/*
 * Point to Point communications
 *
 * Point-to-point communication primitives are used to send and receive arbitrary data between any pair of communicator
 * ranks.
 */

/*
 * Send
 *
 * Sends data from \sendbuff to rank \peer. Rank \peer needs to call hcclRecv with the same \datatype and the same
 * \count from this rank. Blocking for HPU unless operation is part of group within hcclGroupStart() and hcclGroupEnd()
 * section.
 */
hcclResult_t hcclSend(const void*     sendbuff,
                      size_t          count,
                      hcclDataType_t  datatype,
                      int             peer,
                      hcclComm_t      comm,
                      synStreamHandle stream);

/*
 * Receive
 *
 * Receives data from rank \peer to \recvbuff.
 * Rank \peer needs to call hcclSend with the same \datatype and the same \count to this rank.
 * Blocking for HPU unless operation is part of group within hcclGroupStart() and hcclGroupEnd() section.
 */
hcclResult_t
hcclRecv(void* recvbuff, size_t count, hcclDataType_t datatype, int peer, hcclComm_t comm, synStreamHandle stream);

/*
 * Group semantics
 *
 * When managing multiple HPUs from a single thread, and since HCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping HCCL calls as being part of the same collective operation is done
 * using hcclGroupStart and hcclGroupEnd. hcclGroupStart will enqueue all
 * collective calls until the hcclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, hcclGroupEnd only
 * guarantees that the operations are enqueued on the streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and hcclCommInitRank can be used in conjunction
 * of hcclGroupStart/hcclGroupEnd.
 */

/*
 * Group Start
 *
 * Start a group call. All subsequent calls to HCCL may not block due to
 * inter-CPU synchronization.
 */
hcclResult_t hcclGroupStart();

/*
 * Group End
 *
 * End a group call. Wait for all calls since hcclGroupStart to complete
 * before returning.
 */
hcclResult_t hcclGroupEnd();

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // end include guard HCCL_H_
