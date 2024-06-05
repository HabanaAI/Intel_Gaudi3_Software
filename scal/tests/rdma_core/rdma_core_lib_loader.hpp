#pragma once

#include "infiniband/hlibdv.h"

typedef struct ibv_context* (*hlibdv_open_device_fn)(struct ibv_device* device, struct hlibdv_ucontext_attr* attr);
typedef const char* (*hlibv_get_device_name_fn)(struct ibv_device* device);
typedef struct ibv_device** (*hlibv_get_device_list_fn)(int* num_devices);
typedef void (*hlibv_free_device_list_fn)(struct ibv_device** list);
typedef int (*hlibv_close_device_fn)(struct ibv_context* context);

typedef int (*hlibdv_set_port_ex_tmp_fn)(struct ibv_context* context, struct hlibdv_port_ex_attr_tmp* attr);

typedef struct hlibdv_usr_fifo* (*hlibdv_create_usr_fifo_tmp_fn)(struct ibv_context*          context,
                                                             struct hlibdv_usr_fifo_attr_tmp* attr);
typedef int (*hlibdv_destroy_usr_fifo_fn)(struct hlibdv_usr_fifo* usr_fifo);

class IbvLibFunctions
{
public:
    IbvLibFunctions() = default;
    // delete copy constructors
    IbvLibFunctions(const IbvLibFunctions&) = delete;
    IbvLibFunctions& operator= (const IbvLibFunctions&) = delete;
    // move constructor
    IbvLibFunctions(IbvLibFunctions&& src)
    {
        hlib_handle = src.hlib_handle;
        hlibv_get_device_name = src.hlibv_get_device_name;
        hlibv_get_device_list = src.hlibv_get_device_list;
        hlibv_free_device_list = src.hlibv_free_device_list;
        hlibv_close_device = src.hlibv_close_device;
        hlibdv_open_device = src.hlibdv_open_device;
        hlibdv_set_port_ex_tmp = src.hlibdv_set_port_ex_tmp;
        hlibdv_create_usr_fifo_tmp = src.hlibdv_create_usr_fifo_tmp;
        hlibdv_destroy_usr_fifo = src.hlibdv_destroy_usr_fifo;
        src.reset();
    }

    hlibv_get_device_name_fn   hlibv_get_device_name     = nullptr;
    hlibv_get_device_list_fn   hlibv_get_device_list     = nullptr;
    hlibv_free_device_list_fn  hlibv_free_device_list    = nullptr;
    hlibv_close_device_fn      hlibv_close_device        = nullptr;
    hlibdv_open_device_fn      hlibdv_open_device      = nullptr;
    hlibdv_set_port_ex_tmp_fn      hlibdv_set_port_ex_tmp      = nullptr;
    hlibdv_create_usr_fifo_tmp_fn  hlibdv_create_usr_fifo_tmp  = nullptr;
    hlibdv_destroy_usr_fifo_fn hlibdv_destroy_usr_fifo = nullptr;

    bool load();
    void unload();
    bool isLoaded() { return hlib_handle != nullptr; }
    void * libHandle() { return hlib_handle; }
    ~IbvLibFunctions();

private:
    void* hlib_handle = nullptr;

    void reset();
};
