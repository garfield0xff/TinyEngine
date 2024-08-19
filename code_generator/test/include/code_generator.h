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


        // void mean(
        //     uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
        //     uint8_t *output,
        // );


        void clearBuffer();
};

#endif // CODE_GENERATOR_H
