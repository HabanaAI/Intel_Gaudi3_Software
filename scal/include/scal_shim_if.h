#pragma once

#include "scal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SCAL_INTERFACE_VERSION "1.9.0.0"
// ALWAYS ADD NEW FUNCTIONS AT THE END AND UPDATE SCAL_INTERFACE_VERSION !
typedef struct scal_func_table
{
    int (*fp_scal_init)(int fd, const char * config_file_path, scal_handle_t * scal, scal_arc_fw_config_handle_t fwCfg);
    void (*fp_scal_destroy)(const scal_handle_t scal);
    int (*fp_scal_get_fd)(const scal_handle_t scal);
    int (*fp_scal_get_handle_from_fd)(int fd, scal_handle_t* scal);
    uint32_t (*fp_scal_get_sram_size)(const scal_handle_t scal);

    int (*fp_scal_get_pool_handle_by_name)(const scal_handle_t scal, const char *pool_name, scal_pool_handle_t *pool);
    int (*fp_scal_get_pool_handle_by_id)(const scal_handle_t scal, const unsigned pool_id, scal_pool_handle_t *pool);
    int (*fp_scal_pool_get_info)(const scal_pool_handle_t pool, scal_memory_pool_info *info);

    int (*fp_scal_get_core_handle_by_name)(const scal_handle_t scal, const char *core_name, scal_core_handle_t *core);
    int (*fp_scal_get_core_handle_by_id)(const scal_handle_t scal, const unsigned core_id, scal_core_handle_t *core);
    int (*fp_scal_control_core_get_info)(const scal_core_handle_t core, scal_control_core_info_t *info);

    int (*fp_scal_get_stream_handle_by_name)(const scal_handle_t scal, const char * stream_name, scal_stream_handle_t *stream);
    int (*fp_scal_get_stream_handle_by_index)(const scal_core_handle_t scheduler, const unsigned index, scal_stream_handle_t *stream);
    int (*fp_scal_stream_set_commands_buffer)(const scal_stream_handle_t stream, const scal_buffer_handle_t buffer);
    int (*fp_scal_stream_set_priority)(const scal_stream_handle_t stream, const unsigned priority);
    int (*fp_scal_stream_submit)(const scal_stream_handle_t stream, const unsigned pi, const unsigned submission_alignment);
    int (*fp_scal_stream_get_info)(const scal_stream_handle_t stream, scal_stream_info_t *info);

    int (*fp_scal_get_completion_group_handle_by_name)(const scal_handle_t scal, const char * cg_name, scal_comp_group_handle_t *comp_grp);
    int (*fp_scal_get_completion_group_handle_by_index)(const scal_core_handle_t scheduler, const unsigned index, scal_comp_group_handle_t *comp_grp);
    int (*fp_scal_completion_group_wait)(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout);
    int (*fp_scal_completion_group_wait_always_interupt)(const scal_comp_group_handle_t comp_grp, const uint64_t target, const uint64_t timeout);
    int (*fp_scal_completion_group_register_timestamp)(const scal_comp_group_handle_t comp_grp, const uint64_t target, uint64_t timestamps_handle, uint32_t timestamps_offset);
    int (*fp_scal_completion_group_get_info)(const scal_comp_group_handle_t comp_grp, scal_completion_group_info_t *info);

    int (*fp_scal_get_so_pool_handle_by_name)(const scal_handle_t scal, const char *pool_name, scal_so_pool_handle_t *so_pool);
    int (*fp_scal_so_pool_get_info)(const scal_so_pool_handle_t so_pool, scal_so_pool_info *info);

    int (*fp_scal_get_so_monitor_handle_by_name)(const scal_handle_t scal, const char *pool_name, scal_monitor_pool_handle_t *monitor_pool);
    int (*fp_scal_monitor_pool_get_info)(const scal_monitor_pool_handle_t mon_pool, scal_monitor_pool_info *info);

    int (*fp_scal_allocate_buffer)(const scal_pool_handle_t pool, const uint64_t size, scal_buffer_handle_t *buff);
    int (*fp_scal_allocate_aligned_buffer)(const scal_pool_handle_t pool, const uint64_t size, const uint64_t alignment, scal_buffer_handle_t *buff);
    int (*fp_scal_free_buffer)(const scal_buffer_handle_t buff);
    int (*fp_scal_buffer_get_info)(const scal_buffer_handle_t buff, scal_buffer_info_t *info);

    int (*fp_scal_get_cluster_handle_by_name)(const scal_handle_t scal, const char *cluster_name, scal_cluster_handle_t *cluster);
    int (*fp_scal_cluster_get_info)(const scal_cluster_handle_t cluster, scal_cluster_info_t *info);

    uint32_t (*fp_scal_debug_read_reg)(const scal_handle_t scal, uint64_t reg_address);
    int (*fp_scal_debug_write_reg)(const scal_handle_t scal, uint64_t reg_address, uint32_t reg_value);
    int (*fp_scal_debug_memcpy)(const scal_handle_t scal, uint64_t src, uint64_t dst, uint64_t size);
    unsigned (*fp_scal_debug_stream_get_curr_ci)(const scal_stream_handle_t stream);

    int (*fp_scal_control_core_get_debug_info)(const scal_core_handle_t core, uint32_t *arcRegs,
                                               uint32_t arcRegsSize, scal_control_core_debug_info_t *info);
    int (*fp_scal_completion_group_get_infoV2)(const scal_comp_group_handle_t comp_grp, scal_completion_group_infoV2_t *info);
    int (*fp_scal_completion_group_inc_expected_ctr)(scal_comp_group_handle_t comp_grp, uint64_t val);
    int (*fp_scal_completion_group_set_expected_ctr)(scal_comp_group_handle_t comp_grp, uint64_t val);
    int (*fp_scal_bg_work)(const scal_handle_t scal, void (*logFunc)(int, const char*));

    int (*fp_scal_get_sm_info)(const scal_handle_t scal, unsigned sm_idx, scal_sm_info_t *info);

    void (*fp_scal_write_mapped_reg)(volatile uint32_t * pointer, uint32_t value);
    uint32_t (*fp_scal_read_mapped_reg)(volatile uint32_t * pointer);

    int (*fp_scal_get_host_fence_counter_handle_by_name)(const scal_handle_t scal, const char * host_fence_counter_name, scal_host_fence_counter_handle_t *host_fence_counter);
    int (*fp_scal_host_fence_counter_get_info)(scal_host_fence_counter_handle_t host_fence_counter, scal_host_fence_counter_info_t *info);
    int (*fp_scal_host_fence_counter_wait)(const scal_host_fence_counter_handle_t host_fence_counter, const uint64_t num_credits, const uint64_t timeout);
    int (*fp_scal_host_fence_counter_enable_isr)(const scal_host_fence_counter_handle_t host_fence_counter, bool enable_isr);

    int (*fp_scal_get_streamset_handle_by_name)(const scal_handle_t scal, const char *streamset_name, scal_streamset_handle_t *streamset);
    int (*fp_scal_streamset_get_info)(const scal_streamset_handle_t streamset_handle, scal_streamset_info_t* info);

    int (*fp_scal_control_core_get_infoV2)(const scal_core_handle_t core, scal_control_core_infoV2_t *info);

    int (*fp_scal_get_used_sm_base_addrs)(const scal_handle_t scal, unsigned * num_addrs, const scal_sm_base_addr_tuple_t ** sm_base_addr_db);

    int (*fp_scal_debug_background_work)(const scal_handle_t scal);

    int (*fp_scal_pool_get_infoV2)(const scal_pool_handle_t pool, scal_memory_pool_infoV2 *info);

    int (*fp_scal_bg_workV2)(const scal_handle_t scal, void (*logFunc)(int, const char*), char *msg, int msgSize);

    int (*fp_scal_nics_db_fifos_init_and_allocV2)(const scal_handle_t scal, const scal_ibverbs_init_params* ibvInitParams, struct hlibdv_usr_fifo ** createdFifoBuffers, uint32_t * createdFifoBuffersCount);
    // LEAVE PADDING AT THE END OF THE STRUCT - ADD NEW FUNCTIONS ABOVE THIS LINE
    void (*fp_padded_function[10])(void);
} scal_func_table;

#ifdef __cplusplus
}
#endif
