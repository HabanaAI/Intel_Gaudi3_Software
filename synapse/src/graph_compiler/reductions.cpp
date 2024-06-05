#include "reductions.h"

namespace gc::reduction
{
bool datatypeValidForAccumulation(const synDataType datatype)
{
    switch (datatype)
    {
        // This is a list of the non integral types with bit-width < 32
        case syn_type_bf16:
        case syn_type_fp16:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
        case syn_type_ufp16:
            return false;

        // This should be a list of the rest.
        case syn_type_fixed:
        case syn_type_single:
        case syn_type_int16:
        case syn_type_int32:
        case syn_type_uint8:
        case syn_type_int4:
        case syn_type_uint4:
        case syn_type_uint16:
        case syn_type_uint32:
        case syn_type_tf32:
        case syn_type_hb_float:
        case syn_type_int64:
        case syn_type_uint64:
            return true;

        // Default is asserting for invalid types like N/A and 'max' and also new types so they can be added to one of
        // the lists above.
        default:
            HB_ASSERT(false, "Unexpected datatype: {}", datatype);
            return false;
    }
}
}  // namespace gc::reduction