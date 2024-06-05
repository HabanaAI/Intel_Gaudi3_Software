/******************************************************************************
 * Copyright (C) 2020-2022 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************/

//
// This file is included from synapse git in order to be able to call libhcl.so hccl functions - the hccl function (e.g. hcclReduceScatter) is in synapse
// and calls the actual implementation in libhcl.so hcclReduceScatter_impl().
// The calls themselves split to Gen2 / G1 impl functions
// In case new functions are added to hccl.h, this file and its header should be updated and synapse must be rebuilt
//

#include <cstddef>              // for size_t
#include <cstdint>              // for uint64_t

#include "hccl.h"
#include "internal/hccl_impl.h" // for hccl impl
#include "synapse_api_types.h"  // for synStreamHandle
#include "hccl_types.h"         // for hcclResult_t, hcclComm_t, hcclDataType_t

#define HCCL_API_CALL __attribute__((visibility("default")))

hcclResult_t HCCL_API_CALL hcclGetVersion(int* version)
{
    return (hcclGetVersion_impl(version));
}

hcclResult_t HCCL_API_CALL hcclGetUniqueId(hcclUniqueId* uniqueId)
{
    return (hcclGetUniqueId_impl(uniqueId));
}

hcclResult_t HCCL_API_CALL hcclCommInitRank(hcclComm_t* comm, int nranks, hcclUniqueId commId, int rank)
{
    return (hcclCommInitRank_impl(comm, nranks, commId, rank));
}

hcclResult_t HCCL_API_CALL hcclCommInitAll(hcclComm_t* comm, int ndev, const int* devlist)
{
    return (hcclCommInitAll_impl(comm, ndev, devlist));
}

hcclResult_t HCCL_API_CALL hcclCommFinalize(hcclComm_t comm)
{
    return (hcclCommFinalize_impl(comm));
}

hcclResult_t HCCL_API_CALL hcclCommDestroy(hcclComm_t comm)
{
    return (hcclCommDestroy_impl(comm));
}

hcclResult_t HCCL_API_CALL hcclCommAbort(hcclComm_t comm)
{
    return (hcclCommAbort_impl(comm));
}

HCCL_API_CALL const char* hcclGetErrorString(hcclResult_t result)
{
    return (hcclGetErrorString_impl(result));
}

hcclResult_t HCCL_API_CALL hcclCommGetAsyncError(hcclComm_t comm, hcclResult_t* asyncError)
{
    return (hcclCommGetAsyncError_impl(comm, asyncError));
}

hcclResult_t HCCL_API_CALL hcclCommCount(hcclComm_t comm, int* count)
{
    return (hcclCommCount_impl(comm, count));
}

hcclResult_t HCCL_API_CALL hcclCommSynDevice(hcclComm_t comm, int* device)
{
    return (hcclCommSynDevice_impl(comm, device));
}

hcclResult_t HCCL_API_CALL hcclCommUserRank(hcclComm_t comm, int* rank)
{
    return (hcclCommUserRank_impl(comm, rank));
}

int HCCL_API_CALL hcclLookupDMABuff(uint64_t addr, uint64_t size, int* fd)
{
    return (hcclLookupDMABuff_impl(addr, size, fd));
}

hcclResult_t HCCL_API_CALL hcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, hcclDataType_t datatype, hcclRedOp_t reduceOp, hcclComm_t comm, synStreamHandle stream_handle)
{
    return (hcclReduceScatter_impl(sendbuff, recvbuff, recvcount, datatype, reduceOp, comm, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclAllReduce(const void* sendbuff, void* recvbuff, size_t count, hcclDataType_t datatype, hcclRedOp_t reduceOp, hcclComm_t comm, synStreamHandle stream_handle)
{
    return (hcclAllReduce_impl(sendbuff, recvbuff, count, datatype, reduceOp, comm, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclReduce(const void* sendbuff, void* recvbuff, size_t count, hcclDataType_t datatype, hcclRedOp_t reduceOp, int root, hcclComm_t comm,
 synStreamHandle stream_handle)
{
    return (hcclReduce_impl(sendbuff, recvbuff, count, datatype, reduceOp, root, comm, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclBcast(void* buff, size_t count, hcclDataType_t datatype, int root, hcclComm_t comm, synStreamHandle stream_handle)
{
    return (hcclBcast_impl(buff, count, datatype, root, comm, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, hcclDataType_t datatype, int root, hcclComm_t comm_handle, synStreamHandle stream_handle)
{
    return (hcclBroadcast_impl(sendbuff, recvbuff, count, datatype, root, comm_handle, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, hcclDataType_t datatype, hcclComm_t comm_handle, synStreamHandle stream_handle)
{
    return (hcclAllGather_impl(sendbuff, recvbuff, sendcount, datatype, comm_handle, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclAlltoAll(const void* sendbuff, void* recvbuff, size_t count, hcclDataType_t datatype, hcclComm_t comm, synStreamHandle stream_handle)
{
    return (hcclAlltoAll_impl(sendbuff, recvbuff, count, datatype, comm, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclBarrier(hcclComm_t comm_handle, synStreamHandle stream_handle)
{
    return (hcclBarrier_impl(comm_handle, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclSend(const void* sendbuff, size_t count, hcclDataType_t datatype, int peer, hcclComm_t comm_handle, synStreamHandle stream_handle)
{
    return (hcclSend_impl(sendbuff, count, datatype, peer, comm_handle, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclRecv(void* recvbuff, size_t count, hcclDataType_t datatype, int peer, hcclComm_t comm_handle, synStreamHandle stream_handle)
{
    return (hcclRecv_impl(recvbuff, count, datatype, peer, comm_handle, stream_handle));
}

hcclResult_t HCCL_API_CALL hcclGroupStart()
{
    return (hcclGroupStart_impl());
}

hcclResult_t HCCL_API_CALL hcclGroupEnd()
{
    return (hcclGroupEnd_impl());
}
