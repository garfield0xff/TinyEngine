// code_generator.h
#ifndef CODE_GENERATOR_H
#define CODE_GENERATOR_H

#include <cstdint>
#include <iostream>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>

class CodeGenerator {
    private: 
        uint16_t input_x;
        uint16_t input_y;
        uint16_t input_ch;
        float* imageBuffer;
        uint32_t imageBuffer_size;
        
    public:
        CodeGenerator();
        ~CodeGenerator();

        void setImageInputAnd8bitDataBuffer(
            const int image_row_size, const  int image_col_size, const int image_channel,
            const float *imageBuffer, uint32_t imageBuffer_size
        );

        void parseTFModel(const tflite::Model *tf_model);
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
            const int32_t* paddings,  // 패딩 정보를 받는 파라미터
            float *output, const uint16_t output_x, const uint16_t ouptut_y, const uint16_t output_ch,// 결과를 저장할 output 포인터
            int8_t pad_value, int layer_index
        );

        // void mean(
        //     uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
        //     uint8_t *output,
        // );


        void clearBuffer();
};

#endif // CODE_GENERATOR_H
