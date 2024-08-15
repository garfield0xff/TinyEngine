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
        uint8_t* imageBuffer;
        uint32_t imageBuffer_size;
        uint8_t* inputDataBuffer;
        
    public:
        CodeGenerator();
        ~CodeGenerator();

        void setImageInputAnd8bitDataBuffer(
            const int image_row_size, const  int image_col_size, const int image_channel,
            const uint8_t *imageBuffer, uint32_t imageBuffer_size
        );

        void parseTFModel(const tflite::Model *tf_model);
        void conv2d(
            uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            const float *kernel, const float *bias, 
            const float output_activation_min,
            const float output_activation_max, 
            int8_t *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
            uint8_t stride_value
        );

        void depthwiseConv2d(
            int8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            const float *kernel, const float *bias, 
            const float output_activation_min,
            const float output_activation_max, 
            int8_t *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
            uint8_t stride_value
        );

        std::vector<uint8_t> pad2d(
            uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            const float *kernel, const float *bias, 
            int8_t *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
            int8_t pad_value
        );


        void clearBuffer();
};

#endif // CODE_GENERATOR_H
