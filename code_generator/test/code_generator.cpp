// code_generator.cpp
#include "code_generator.h"
#include <algorithm>

CodeGenerator::CodeGenerator() : imageBuffer(nullptr), imageBuffer_size(0) {}

CodeGenerator::~CodeGenerator() {
    clearBuffer();
}

void CodeGenerator::setImageInputAnd8bitDataBuffer(
    const int image_row_size, const int image_col_size, const int image_channel, 
    const uint8_t* _imageBuffer, uint16_t _imageBuffer_size
)
{
    this->input_y = static_cast<uint16_t>(image_row_size);
    this->input_x = static_cast<uint16_t>(image_col_size);
    this->input_ch = static_cast<uint16_t>(image_channel);
    this->imageBuffer_size = _imageBuffer_size;

    delete[] this->imageBuffer;
    this->imageBuffer = new uint8_t[_imageBuffer_size];
    std::copy(_imageBuffer, _imageBuffer + _imageBuffer_size, this->imageBuffer);

}

/*
    parseModel to get tensor info
    int8_t kernel, int32_t bias, int8_t pad uint8_t stride
    uint16_t output_x, uint16_t output_y, uint16_t output_ch
*/

void CodeGenerator::parseTFModel(const tflite::Model *tf_model)
{
    if(tf_model->subgraphs()->size() == 0)
    {
        std::cerr << "Model has no graph" << "\n";
        return;
    }

    // Get tensorflow model backbone
    const tflite::SubGraph* subgraph = tf_model->subgraphs()->Get(0);

    bool flag = false;
    for(size_t i = 0; !flag && i < subgraph->operators()->size(); ++i)
    {
        // operation index
        auto op = subgraph->operators()->Get(i);
        auto opcode_index = op->opcode_index();
        auto opcode = tf_model->operator_codes()->Get(opcode_index)->builtin_code();
        

        
        // switch by operator type ex ) CONV2D ,DEPTHWISE_CONV2D
        switch (opcode)
        {
        case tflite::BuiltinOperator_CONV_2D:
        {
            // get convolution info
            const tflite::Conv2DOptions* conv_options = op->builtin_options_as_Conv2DOptions();
            uint8_t stride = conv_options-> stride_h();
            
            // get kernel info
            auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
            auto filter_shape = filter_tensor->shape();
            auto filter_type = filter_tensor->type();

            auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer());
            size_t buffer_size = filter_buffer->data()->size() / sizeof(float);
            std::cout << "buffer size is : "<< buffer_size << std::endl;
            std::vector<float> flat_filter_buffer(buffer_size);

            // // kernel_buffer to  flat kernel buffer
            std::memcpy(flat_filter_buffer.data(), filter_buffer->data()->data(), filter_buffer->data()->size());

            
            flag = true;
        } 
            
            break;
        
        default:
            break;
        }
    }

}

void CodeGenerator::clearBuffer() {
    if (imageBuffer != nullptr) {
        delete[] imageBuffer;
        imageBuffer = nullptr;
    }
}

void CodeGenerator::conv2d(
    const uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const int8_t *kernel, const int32_t *bias,   const int32_t *biasR, const float *scales,
    const int32_t output_offset, const int32_t input_offset,
    const int32_t output_activation_min,
    const int32_t output_activation_max, 
    int8_t *output, const int16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    int8_t pad_value
)
{
    
}