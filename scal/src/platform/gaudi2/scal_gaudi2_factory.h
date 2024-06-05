#pragma once
#include "hlthunk.h"
#include "scal_base.h"
Scal * create_scal_gaudi2(const int fd, const struct hlthunk_hw_ip_info & hw_ip, scal_arc_fw_config_handle_t fwCfg);