#ifndef CODE_VALIDATOR_H
#define CODE_VALIDATOR_H

#include "engine.h"
#include <iostream>
#include <fstream>
#include <vector>


class Validator : public Engine {

    private:
        std::string layer_path;

    public:

        void set_layer_folder_path(const std::string _layer_path) {
            layer_path = _layer_path;
        }
        
        void conv2d(
            float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            const float *kernel, size_t kernelSize, int kernel_height, int kernel_width,
            const float *bias, 
            const float output_activation_min,
            const float output_activation_max, 
            float *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
            uint8_t stride_value, int layer_index, bool relu_flag, bool pad_is_valid
        );


        void depthwiseConv2d(
            float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            const float *kernel, size_t kernel_size, int kernel_height, int kernel_width,
            const float *bias,
            const float output_activation_min,
            const float output_activation_max, 
            float *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
            uint8_t stride_value, int layer_index, bool relu_flag, bool pad_is_valid
        );


        void pad2d(
            float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            const int32_t* paddings,  
            float *output, const uint16_t output_x, const uint16_t ouptut_y, const uint16_t output_ch,// 결과를 저장할 output 포인터
            int8_t pad_value, int layer_index
        );

        void add(
            float* input_tensor1, float* input_tensor2, 
            const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            float* output, int layer_index
        );

};


#endif