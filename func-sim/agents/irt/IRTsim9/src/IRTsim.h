/*
 * IRTsim.h
 *
 *  Created on: Jun 4, 2019
 *      Author: espektor
 */

#ifndef IRTSIM_H_
#define IRTSIM_H_

#include "IRT.h"
namespace irt_utils_gaudi3
{
extern void ext_memory_wrapper(unsigned long long int addr,
                               bool                   read,
                               bool                   write,
                               bus128B_struct         wr_data,
                               bus128B_struct&        rd_data,
                               meta_data_struct       meta_in,
                               meta_data_struct&      meta_out,
                               int                    lsb_pad,
                               int                    msb_pad);

extern FILE* test_res;

void IRThl(IRT_top* irt_top, int image, int desc, int i_width, int i_height, char* file_name);
} // namespace irt_utils_gaudi3

#endif /* IRTSIM_H_ */
