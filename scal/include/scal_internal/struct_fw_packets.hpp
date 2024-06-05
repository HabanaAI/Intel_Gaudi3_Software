#pragma once

// we include the packets (eng and arc) into different structs to avoid name collisions between
// Gaudi2 and Gaudi3.
// It means that code that wants those headers must include them from here

#ifdef __GAUDI2_ARC_SCHED_PACKETS_H__
#error "gaudi2_arc_sched_packets.h is already included. please remove the direct inclusion of gaudi2_arc_sched_packets.h from your code"
#endif

#ifdef __GAUDI2_ARC_HOST_PACKETS_H__
#error "gaudi2_arc_host_packets.h is already included. please remove the direct inclusion of gaudi2_arc_host_packets.h from your code"
#endif

#ifdef __GAUDI3_ARC_SCHED_PACKETS_H__
#error "gaudi3_arc_sched_packets.h is already included. please remove the direct inclusion of gaudi3_arc_sched_packets.h from your code"
#endif

#ifdef __GAUDI3_ARC_HOST_PACKETS_H__
#error "gaudi3_arc_host_packets.h is already included. please remove the direct inclusion of gaudi3_arc_host_packets.h from your code"
#endif

struct g2fw
{
#include "gaudi2_arc_sched_packets.h"
#include "gaudi2_arc_host_packets.h"
    struct sched_arc_cmd_acp_fence_wait_t{};
    // packetToName requires a unique ID (value) for each packet type (for any platform) => Count+1
    static const uint8_t SCHED_COMPUTE_ARC_CMD_ACP_FENCE_WAIT = (SCHED_COMPUTE_ARC_CMD_COUNT + 1);

    struct sched_arc_cmd_acp_fence_inc_immediate_t{};
    // packetToName requires a unique ID (value) for each packet type (for any platform) => Count+1
    static const uint8_t SCHED_COMPUTE_ARC_CMD_ACP_FENCE_INC_IMMEDIATE = (SCHED_COMPUTE_ARC_CMD_COUNT + 2);

    struct sched_arc_cmd_dispatch_cme_ecb_list_t{};
};

struct g3fw
{
#include "gaudi3/gaudi3_arc_sched_packets.h"
#include "gaudi3/gaudi3_arc_host_packets.h"
};
