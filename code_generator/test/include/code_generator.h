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
        uint16_t imageBuffer_size;
        
    public:
        CodeGenerator();
        ~CodeGenerator();

        void setImageInputAnd8bitDataBuffer(
            const int image_row_size, const  int image_col_size, const int image_channel,
            const uint8_t *imageBuffer, uint16_t imageBuffer_size
        );

        void parseTFModel(const tflite::Model *tf_model);
        void conv2d(
            const uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
            const int8_t *kernel, const int32_t *bias,   const int32_t *biasR, const float *scales,
            const int32_t output_offset, const int32_t input_offset,
            const int32_t output_activation_min,
            const int32_t output_activation_max, 
            int8_t *output, const int16_t output_x, const uint16_t output_y, const uint16_t output_ch,
            int8_t pad_value
        );

        void clearBuffer();
};

#endif // CODE_GENERATOR_H
