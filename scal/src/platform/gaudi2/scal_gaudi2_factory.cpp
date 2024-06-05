#include "scal_gaudi2_factory.h"
#include "scal_gaudi2.h"

Scal * create_scal_gaudi2(const int fd, const struct hlthunk_hw_ip_info & hw_ip, scal_arc_fw_config_handle_t fwCfg)
{
    return new Scal_Gaudi2(fd, hw_ip, fwCfg);
}