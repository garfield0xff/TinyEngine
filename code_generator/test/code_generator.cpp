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

    // Store convolution result data buffer
    int16_t x = 0;
    int16_t y = 0;
    int16_t ch = 0;
    int8_t* data_buffer = imageBuffer; 

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
            data_buffer = new int8_t[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];
            std::cout << "iamgeBuffer_index is : " << imageBuffer_size << std::endl;
            std::cout << "image buffer in last index : " << static_cast<int>(imageBuffer[imageBuffer_size - 1]) << std::endl;

            x = output_shape->Get(1);
            y = output_shape->Get(2);
            ch = output_shape->Get(3);

            conv2d(
                inputDataBuffer, input_x, input_y, input_ch,
                flat_filter_buffer.data(), flat_bias_buffer.data(),
                0.0, 0.0,
                data_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                stride
            );

            // depthwiseConv2d(imageBuffer, input_x, input_y, input_ch,
            //     flat_filter_buffer.data(), flat_bias_buffer.data(),
            //     0.0, 0.0,
            //     output_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
            //     stride
            // );

            // pad2d(
            //     imageBuffer, input_x, input_y, input_ch, 
            //     flat_filter_buffer.data(), flat_bias_buffer.data(),
            //     output_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
            //     0
            // );
        }     
        break;

        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        {
            const tflite::DepthwiseConv2DOptions* conv_options = op->builtin_options_as_DepthwiseConv2DOptions();
            uint8_t stride = conv_options-> stride_h();

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

            depthwiseConv2d(
                data_buffer, x, y, ch,
                flat_filter_buffer.data(), flat_bias_buffer.data(),
                0.0, 0.0,
                output_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                stride
            );

            flag = true;
        }

        
        
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
    const float output_activation_min,
    const float output_activation_max, 
    int8_t *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    uint8_t stride_value
)
{
    // Output dimensions
    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    const int kernel_size = 3;

    for (int oc = 0; oc < output_channels; ++oc) { 
        for (int oy = 0; oy < output_height; ++oy) { 
            for (int ox = 0; ox < output_width; ++ox) { 
                float acc = 0.0f;
                for (int ic = 0; ic < input_ch; ++ic) { 
                    for (int ky = 0; ky < kernel_size; ++ky) { 
                        for (int kx = 0; kx < kernel_size; ++kx) { 
                            int ix = ox * stride_value + kx;
                            int iy = oy * stride_value + ky;
                            if (ix < input_x && iy < input_y) {
                                int input_index = (iy * input_x + ix) * input_ch + ic;
                                int kernel_index = ((oc * kernel_size * kernel_size * input_ch) +
                                                    (ky * kernel_size * input_ch) +
                                                    (kx * input_ch) + ic);
                                acc += input[input_index] * kernel[kernel_index];
                            }
                        }
                    }
                }
                // Add bias and apply ReLU6
                acc += bias[oc];
                acc = std::max(0.0f, std::min(acc, 6.0f)); // ReLU6

                // Store the result in the output tensor (convert float to int8_t)
                int output_index = (oy * output_x + ox) * output_channels + oc;
                output[output_index] = static_cast<int8_t>(std::max(std::min(acc, 127.0f), -128.0f)); // Clamp to int8_t range
            }
        }
    }

    // Write output tensor to file
    std::ofstream out_file("outputConv2d.txt");
    if (out_file.is_open()) {
        for (int oc = 0; oc < output_channels; ++oc) {
            out_file << "Output Channel " << oc + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_height; ++oy) {
                for (int ox = 0; ox < output_width; ++ox) {
                    int output_index = (oy * output_x + ox) * output_channels + oc;
                    out_file << static_cast<int>(output[output_index]) << " ";
                }
                out_file << std::endl;
            }
            out_file << std::endl;
        }
        out_file.close(); // Close the file after writing
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
}

void CodeGenerator::depthwiseConv2d(
    int8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const float *kernel, const float *bias, 
    const float output_activation_min,
    const float output_activation_max, 
    int8_t *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    uint8_t stride_value
)
{
    // Output dimensions
    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    const int kernel_size = 3;

    // Depthwise convolution operation
    for (int ic = 0; ic < input_ch; ++ic) {  // Input (and output) channels loop
        for (int oy = 0; oy < output_height; ++oy) {  // Output height loop
            for (int ox = 0; ox < output_width; ++ox) {  // Output width loop
                float acc = 0.0f;
                for (int ky = 0; ky < kernel_size; ++ky) {  // Kernel height loop
                    for (int kx = 0; kx < kernel_size; ++kx) {  // Kernel width loop
                        int ix = ox * stride_value + kx;
                        int iy = oy * stride_value + ky;
                        if (ix < input_x && iy < input_y) {
                            int input_index = (iy * input_x + ix) * input_ch + ic;
                            int kernel_index = (ic * kernel_size * kernel_size) + (ky * kernel_size + kx);
                            acc += input[input_index] * kernel[kernel_index];
                        }
                    }
                }
                // Add bias and apply ReLU6
                acc += bias[ic];
                acc = std::max(0.0f, std::min(acc, 6.0f));  // ReLU6

                // Store the result in the output tensor (convert float to int8_t)
                int output_index = (oy * output_x + ox) * output_channels + ic;
                output[output_index] = static_cast<int8_t>(std::max(std::min(acc, 127.0f), -128.0f));  // Clamp to int8_t range
            }
        }
    }

    // Write output tensor to file
    std::ofstream out_file("outputDepthwiseConv2d.txt");
    if (out_file.is_open()) {
        for (int ic = 0; ic < input_ch; ++ic) {
            out_file << "Output Channel " << ic + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_height; ++oy) {
                for (int ox = 0; ox < output_width; ++ox) {
                    int output_index = (oy * output_x + ox) * output_channels + ic;
                    out_file << static_cast<int>(output[output_index]) << " ";
                }
                out_file << std::endl;
            }
            out_file << std::endl;
        }
        out_file.close();  // Close the file after writing
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
}


std::vector<uint8_t> CodeGenerator::pad2d(
    uint8_t *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const float *kernel, const float *bias, 
    int8_t *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
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

    return padded_input;
}   