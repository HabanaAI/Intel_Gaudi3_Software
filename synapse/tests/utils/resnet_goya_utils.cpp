#include "resnet_goya_utils.h"
#include <string.h>
#include <limits>
// return index of max element
int max_index(int8_t* in, int size)
{
    int idx, i;
    int8_t max;

    if (!size)
    {
        return -1;
    }

    idx = 0;
    max = in[0];

    for (i =1 ; i < size; ++i)
    {
        if( in[i] > max )
        {
            max = in[i];
            idx = i;
        }
    }

    return idx;
}

// return indexes of N top elements
void top(int8_t* in, int size, int* out, int N)
{
    int i, idx;
    int8_t *tmp;

    tmp = (int8_t*) malloc( size*sizeof(int8_t) );
    memcpy( tmp, in, size*sizeof(int8_t) );

    for (i = 0; i < N; ++i )
    {
        idx = max_index(tmp,size);
        out[i]=idx;
        tmp[idx] = std::numeric_limits<int8_t>::min();
    }
    free(tmp);
}
