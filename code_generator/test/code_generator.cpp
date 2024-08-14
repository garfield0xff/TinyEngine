// code_generator.cpp
#include "code_generator.h"
#include <algorithm>
#include <fstream>

CodeGenerator::CodeGenerator() : imageBuffer(nullptr), imageBuffer_size(0) {}

CodeGenerator::~CodeGenerator() {
    clearBuffer();
}

void CodeGenerator::setImageInputAnd8bitDataBuffer(
    const int image_row_size, const int image_col_size, const int image_channel, 
    const uint8_t* _imageBuffer, uint32_t _imageBuffer_size
)
{
    this->input_y = static_cast<uint16_t>(image_row_size);
    this->input_x = static_cast<uint16_t>(image_col_size);
    this->input_ch = static_cast<uint16_t>(image_channel);
    this->imageBuffer_size = _imageBuffer_size;

    delete[] this->imageBuffer;
    this->imageBuffer = new uint8_t[_imageBuffer_size];
    std::copy(_imageBuffer, _imageBuffer + _imageBuffer_size, this->imageBuffer);

    std::cout << "image buffer is :  " << static_cast<int>(this->imageBuffer[imageBuffer_size - 1])  << std::endl;

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
            size_t filter_buffer_size = filter_buffer->data()->size() / sizeof(float);
            std::cout << "buffer size is : "<< filter_buffer_size << std::endl;
            std::vector<float> flat_filter_buffer(filter_buffer_size);

            // get  bias info
            auto bias_tensor = subgraph->tensors()->Get(op->inputs()->Get(2));
            auto bias_shape = bias_tensor->shape();
            auto bias_type = bias_tensor->type();

            auto bias_buffer = tf_model->buffers()->Get(bias_tensor->buffer());
            size_t bias_buffer_size = bias_buffer->data()->size() / sizeof(float);
            std::vector<float> flat_bias_buffer(bias_buffer_size);
            

            // // kernel_buffer to  flat kernel buffer
            std::memcpy(flat_filter_buffer.data(), filter_buffer->data()->data(), filter_buffer->data()->size());
            std::memcpy(flat_bias_buffer.data(), bias_buffer->data()->data(), bias_buffer->data()->size());
            
            
            auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
            // [-1, h, w, c]
            auto output_shape = output_tensor->shape();
        
            std::cout << "output buffer size is : "<< filter_buffer_size << std::endl;

            // temp output buffer -> 값 계산해보고 타입 변환 해줘야됌!
            int8_t output_buffer[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];
            std::cout << "iamgeBuffer_index is : " << imageBuffer_size << std::endl;
            std::cout << "image buffer in last index : " << static_cast<int>(imageBuffer[imageBuffer_size - 1]) << std::endl;

            conv2d(
                imageBuffer, input_x, input_y, input_ch, 
                flat_filter_buffer.data(), flat_bias_buffer.data(),
                6.0, 0.0, 
                output_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                0
            );
            
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

void print_padded_channel_to_file(std::ofstream &out_file, uint8_t *data, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            out_file << static_cast<int>(data[i * width + j]) << " ";
        }
        out_file << std::endl;
    }
}

void CodeGenerator::conv2d(
    uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const float *kernel, const float *bias, 
    const int32_t output_activation_min,
    const int32_t output_activation_max, 
    int8_t *output, const int16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    int8_t pad_value
)
{
    // Allocate memory for padded input data
    std::vector<uint8_t> padded_input((input_x + 2) * (input_y + 2) * input_ch, pad_value);

    // Set up pointers for copying data
    uint8_t *padded_ptr = padded_input.data();
    uint8_t *input_ptr = input;

    // Copy input data into padded_input with padding
    for (int c = 0; c < input_ch; ++c) {
        padded_ptr += (input_x + 2); // Skip top padding
        for (int i = 0; i < input_y; ++i) {
            *padded_ptr++ = pad_value; // Left padding
            std::memcpy(padded_ptr, input_ptr, input_x); // Copy row data
            padded_ptr += input_x;
            *padded_ptr++ = pad_value; // Right padding
            input_ptr += input_x;
        }
        padded_ptr += (input_x + 2); // Skip bottom padding
    }

    // Open the file to save the results
    std::ofstream out_file("test.txt");

    if (out_file.is_open()) {
        // Write padded input data to file
        padded_ptr = padded_input.data();  // Reset pointer to start of padded input data
        for (int c = 0; c < input_ch; ++c) {
            out_file << "Channel " << c + 1 << ":" << std::endl;
            print_padded_channel_to_file(out_file, padded_ptr, input_x + 2, input_y + 2);
            padded_ptr += (input_x + 2) * (input_y + 2);  // Move to next channel
            out_file << std::endl;
        }

        out_file.close(); // Close the file after writing
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
}