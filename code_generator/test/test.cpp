#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

const int TILE_SIZE = 8;

void depthwise_kernel3x3_stride1 (
    const uint16_t output_y, const uint16_t output_x,
    const int32_t *bias, const int32_t *biasR, const int8_t *ksrc, const float *scales,
    int8_t *output, int8_t* *output_mask, const int mask_idx, const int32_t output_offset,
    const int32_t act_min, const int32_t act_max,
    int8_t *cols_8b_iterptr, const uint16_t column_x, int channel_offset
)
{
    #define STRIDE 1
    int i, j;
    int8_t mask_value;

    for (i = 0; i < output_y; i++) {
        for(j = 0; j < output_x; j++) {
            int8_t *cols_8b
        }
    }
}
int main()
{
    
}