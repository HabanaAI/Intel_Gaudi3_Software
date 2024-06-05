#include <stdio.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <string>
#include "rdma_core_lib_loader.hpp"
#include "logger.h"
#include "../../src/common/scal_macros.h"

#define _SO_FUNC_(handle, fname)                                                                                       \
    fname = (fname##_fn)dlsym(handle, #fname);                                                                         \
    if (!fname) return false

#define LIBFUNC(fname) _SO_FUNC_(hlib_handle, fname)

bool IbvLibFunctions::load()
{
    if (hlib_handle != nullptr) return false;

    const char *env_var = getenv("RDMA_CORE_LIB");
    scal_assert_return(env_var != 0, false, "failed to get RDMA_CORE_LIB");

    /* get handle for libibverbs */
    dlerror(); // clear dlerror

    /* get handle for libhlib with the same namespace as libibverbs */
    dlerror(); // clear dlerror
    std::string libhlibstr(env_var);
    libhlibstr.append("/libhlib.so");
    hlib_handle = dlopen(libhlibstr.c_str(), RTLD_NOW | RTLD_LOCAL);
    scal_assert_return(hlib_handle != 0, false, "libhlib.so load failed path={} errno={} error={} dlerror={}", libhlibstr, errno, std::strerror(errno), dlerror());

    /* get necessary function pointers */
    LIBFUNC(hlibdv_open_device);
    LIBFUNC(hlibdv_set_port_ex_tmp);
    LIBFUNC(hlibdv_create_usr_fifo_tmp);
    LIBFUNC(hlibdv_destroy_usr_fifo);

    LIBFUNC(hlibv_get_device_list);
    LIBFUNC(hlibv_get_device_name);
    LIBFUNC(hlibv_free_device_list);
    LIBFUNC(hlibv_close_device);

    return true;
}


void IbvLibFunctions::reset()
{
    hlib_handle = nullptr;
    hlibv_get_device_name = nullptr;
    hlibv_get_device_list = nullptr;
    hlibv_free_device_list = nullptr;
    hlibv_close_device = nullptr;
    hlibdv_open_device = nullptr;
    hlibdv_set_port_ex_tmp = nullptr;
    hlibdv_create_usr_fifo_tmp = nullptr;
    hlibdv_destroy_usr_fifo = nullptr;
}

void IbvLibFunctions::unload()
{
    if (hlib_handle) dlclose(hlib_handle);
    reset();
}

IbvLibFunctions::~IbvLibFunctions()
{
    unload();
}
