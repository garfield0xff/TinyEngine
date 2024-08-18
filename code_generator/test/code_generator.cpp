// code_generator.cpp
#include "code_generator.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;

const std::string layer_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/test/layer/";

CodeGenerator::CodeGenerator() : imageBuffer(nullptr), imageBuffer_size(0) {}

CodeGenerator::~CodeGenerator() {
    clearBuffer();
}

void CodeGenerator::setImageInputAnd8bitDataBuffer(
    const int image_row_size, const int image_col_size, const int image_channel, 
    const float* _imageBuffer, uint32_t _imageBuffer_size
)
{
    this->input_y = static_cast<uint16_t>(image_row_size);
    this->input_x = static_cast<uint16_t>(image_col_size);
    this->input_ch = static_cast<uint16_t>(image_channel);
    this->imageBuffer_size = _imageBuffer_size;

    delete[] this->imageBuffer;
    this->imageBuffer = new float[_imageBuffer_size];
    std::copy(_imageBuffer, _imageBuffer + _imageBuffer_size, this->imageBuffer);

    std::ofstream out_file(layer_path + "normalized_image_data.txt");

    if (out_file.is_open()) {
        for (int c = 0; c < input_ch; ++c) {
            out_file << "Channel " << c + 1 << ":" << std::endl;

            for (int y = 0; y < input_y; ++y) {
                for (int x = 0; x < input_x; ++x) {
                    int index = (y * input_x + x) * input_ch + c;
                    out_file << static_cast<int>(imageBuffer[index]) << " ";
                }
                out_file << std::endl;
            }

            out_file << std::endl;  
        }

        out_file.close(); // 파일을 닫습니다.
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }

    for(int i = 0; i < 10; i ++)
    {
        std::cout << "flat imgae buffer is :" << _imageBuffer[i] << " ";
    }
    std::cout << std::endl;
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
    int16_t x = input_x;
    int16_t y = input_y;
    int16_t ch = input_ch;
    float* input_data_buffer = imageBuffer; 
    float* output_data_buffer = nullptr; 
    float* temp_add_data_buffer = nullptr;
    float* mean_data_buffer = nullptr;
    // float* fully_connected_data_buffer = nullptr;
    float temp_sum = 0;

    bool flag = false;
    int func_count = 0;

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

            std::cout << "operator : " << i << " Convolution" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
            // get convolution info
            const tflite::Conv2DOptions* conv_options = op->builtin_options_as_Conv2DOptions();
            tflite::ActivationFunctionType activation_type = conv_options->fused_activation_function();
            uint8_t stride = conv_options-> stride_h();

            bool relu_flag = false;
            bool pad_is_valid = true;

            if (activation_type == tflite::ActivationFunctionType::ActivationFunctionType_RELU) {
                std::cout << "ReLU activation is included in this convolution." << std::endl;
            } else if (activation_type == tflite::ActivationFunctionType::ActivationFunctionType_RELU6) {
                std::cout << "ReLU6 activation is included in this convolution." << std::endl;
                relu_flag = true;
            } else {
                std::cout << "No ReLU or ReLU6 activation is included in this convolution." << std::endl;
            }

            if (conv_options->padding() == tflite::Padding::Padding_VALID) {
                std::cout << "Padding is set to VALID." << std::endl;
                pad_is_valid = false;
            } else if (conv_options->padding() == tflite::Padding::Padding_SAME) {
                std::cout << "Padding is set to SAME." << std::endl;
                pad_is_valid = true;
            } else {
                std::cout << "Unknown padding type." << std::endl;
            }

            
            // get kernel info
            auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
            auto filter_shape = filter_tensor->shape();
            auto filter_type = filter_tensor->type();

            auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer());
            const float* filter_data = reinterpret_cast<const float*>(filter_buffer->data()->data());
            size_t filter_buffer_size = filter_buffer->data()->size() / sizeof(float);
            std::cout << "buffer size is : "<< filter_buffer_size << std::endl;
            std::vector<float> flat_filter_buffer(filter_buffer_size);

            int filter_height = filter_shape->Get(1);
            int filter_width = filter_shape->Get(2);
            
            
            // get  bias info
            auto bias_tensor = subgraph->tensors()->Get(op->inputs()->Get(2));
            auto bias_shape = bias_tensor->shape();
            auto bias_type = bias_tensor->type();

            auto bias_buffer = tf_model->buffers()->Get(bias_tensor->buffer());
            const float* bias_data = reinterpret_cast<const float*>(bias_buffer->data()->data());
            size_t bias_buffer_size = bias_buffer->data()->size() / sizeof(float);
            std::vector<float> flat_bias_buffer(bias_buffer_size);
            
            // // kernel_buffer to  flat kernel buffer
            std::copy(filter_data, filter_data + filter_buffer_size, flat_filter_buffer.begin());
            std::copy(bias_data, bias_data + bias_buffer_size, flat_bias_buffer.begin());
            // std::memcpy(flat_bias_buffer.data(), bias_buffer->data()->data(), bias_buffer->data()->size());

            std::cout << "flat filter buffer first index " <<  i <<  " is  : "  << flat_filter_buffer[0] << std::endl;
            std::cout << "bias buffer first index " <<  i <<  " is  : "  << flat_bias_buffer[0] << std::endl;
            
            
            auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
            // [-1, h, w, c]
            auto output_shape = output_tensor->shape();

            std::cout << "input_data_buffer is : " ;
            for(int j = 0; j < 10; j++)
            {
                std::cout << input_data_buffer[j] << " ";
            }
        
            // temp output buffer -> 값 계산해보고 타입 변환 해줘야됌!
            delete[] output_data_buffer;
            output_data_buffer = new float[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];
            std::cout << "output buffer size is : "<< output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3) << std::endl;

            conv2d(
                input_data_buffer, x, y, ch,
                flat_filter_buffer.data(), filter_buffer_size, filter_height, filter_width,
                flat_bias_buffer.data(), 
                0.0, 6.0,
                output_data_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                stride, i, relu_flag, pad_is_valid
            );

            x = output_shape->Get(1);
            y = output_shape->Get(2);
            ch = output_shape->Get(3);

            std::cout << "output_data buffer is : " ;
            for(int j = 0; j < 10; j++)
            {
                std::cout << output_data_buffer[j] << " ";
            }
            std::cout << std::endl;

            // input_data_buffer = output_data_buffer;
            input_data_buffer = std::move(output_data_buffer);

            std::cout << std::endl;

            for (size_t j = i + 1; j < subgraph->operators()->size(); ++j)
            {
                auto next_op = subgraph->operators()->Get(j);
                auto next_opcode_index = next_op->opcode_index();
                auto next_opcode = tf_model->operator_codes()->Get(next_opcode_index)->builtin_code();

                if (next_opcode == tflite::BuiltinOperator_ADD)
                {
                    auto add_inputs = next_op->inputs();
                    for (size_t k = 0; k < add_inputs->size(); ++k)
                    {
                        if (add_inputs->Get(k) == op->outputs()->Get(0))
                        {
                            temp_add_data_buffer = output_data_buffer;
                            std::cout << "This tensor is used in an ADD operation." << std::endl;
                            std::cout << "1.temp indexing .. " << std::endl;
                            for(size_t t = 0; t < 10; ++t)
                            {
                                std::cout << temp_add_data_buffer[t] <<  " ";
                            }
                            std::cout << std::endl;

                            break;
                        }
                    }
                    
                }
            }
            std::cout << "------------------------------------------------" << std::endl;

            
            func_count++;
        }     
        break;

        case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        {
            std::cout << "operator : " << i << "  DepthWise Convolution" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
            
            const tflite::DepthwiseConv2DOptions* conv_options = op->builtin_options_as_DepthwiseConv2DOptions();
            tflite::ActivationFunctionType activation_type = conv_options->fused_activation_function();
            uint8_t stride = conv_options-> stride_h();

            bool relu_flag = false;
            bool pad_is_valid = true;

            if (activation_type == tflite::ActivationFunctionType::ActivationFunctionType_RELU) {
                std::cout << "ReLU activation is included in this convolution." << std::endl;
            } else if (activation_type == tflite::ActivationFunctionType::ActivationFunctionType_RELU6) {
                std::cout << "ReLU6 activation is included in this convolution." << std::endl;
                relu_flag = true;
            } else {
                std::cout << "No ReLU or ReLU6 activation is included in this convolution." << std::endl;
            }

            if (conv_options->padding() == tflite::Padding::Padding_VALID) {
                std::cout << "Padding is set to VALID." << std::endl;
                pad_is_valid = false;
            } else if (conv_options->padding() == tflite::Padding::Padding_SAME) {
                std::cout << "Padding is set to SAME." << std::endl;
                pad_is_valid = true;
            } else {
                std::cout << "Unknown padding type." << std::endl;
            }

            auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
            auto filter_shape = filter_tensor->shape();
            auto filter_type = filter_tensor->type();

            auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer());
            const float* filter_data = reinterpret_cast<const float*>(filter_buffer->data()->data());
            size_t filter_buffer_size = filter_buffer->data()->size() / sizeof(float);
            std::cout << "buffer size is : "<< filter_buffer_size << std::endl;
            std::vector<float> flat_filter_buffer(filter_buffer_size);

            // get  bias info
            auto bias_tensor = subgraph->tensors()->Get(op->inputs()->Get(2));
            auto bias_shape = bias_tensor->shape();
            auto bias_type = bias_tensor->type();

            auto bias_buffer = tf_model->buffers()->Get(bias_tensor->buffer());
            const float* bias_data = reinterpret_cast<const float*>(bias_buffer->data()->data());
            size_t bias_buffer_size = bias_buffer->data()->size() / sizeof(float);
            std::vector<float> flat_bias_buffer(bias_buffer_size);
            

            // // kernel_buffer to  flat kernel buffer
            std::copy(filter_data, filter_data + filter_buffer_size, flat_filter_buffer.begin());
            std::copy(bias_data, bias_data + bias_buffer_size, flat_bias_buffer.begin());
            // std::memcpy(flat_bias_buffer.data(), bias_buffer->data()->data(), bias_buffer->data()->size());

            std::cout << "flat filter buffer first index " <<  i <<  " is  : "  << flat_filter_buffer[0] << std::endl;
            std::cout << "bias buffer first index " <<  i <<  " is  : "  << flat_bias_buffer[0] << std::endl;
            
            auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
            // [-1, h, w, c]
            auto output_shape = output_tensor->shape();
        
            int filter_height = filter_shape->Get(1);
            int filter_width = filter_shape->Get(2);

            // temp output buffer -> 값 계산해보고 타입 변환 해줘야됌!
            delete[] output_data_buffer;
            output_data_buffer = new float[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];
            
            std::cout << "output buffer size is : "<< output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3) << std::endl;

            depthwiseConv2d(
                input_data_buffer, x, y, ch,
                flat_filter_buffer.data(), flat_bias_buffer.data(), filter_buffer_size, filter_height, filter_width, 
                0.0, 6.0,
                output_data_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                stride, i, relu_flag, pad_is_valid
            );

            x = output_shape->Get(1);
            y = output_shape->Get(2);
            ch = output_shape->Get(3);

            // input_data_buffer = output_data_buffer;
            input_data_buffer = std::move(output_data_buffer);

            for (size_t j = i + 1; j < subgraph->operators()->size(); ++j)
            {
                auto next_op = subgraph->operators()->Get(j);
                auto next_opcode_index = next_op->opcode_index();
                auto next_opcode = tf_model->operator_codes()->Get(next_opcode_index)->builtin_code();

                if (next_opcode == tflite::BuiltinOperator_ADD)
                {
                    auto add_inputs = next_op->inputs();
                    for (size_t k = 0; k < add_inputs->size(); ++k)
                    {
                        if (add_inputs->Get(k) == op->outputs()->Get(0))
                        {
                            
                            temp_add_data_buffer = output_data_buffer;
                            std::cout << "This tensor is used in an ADD operation." << std::endl;
                            break;
                        }
                    }
                }
            }

            std::cout << "------------------------------------------------" << std::endl;

            func_count++;
        }
        break;

        case tflite::BuiltinOperator_PAD:
        {
            std::cout << "operator : " << i << "  PAD" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
            auto paddings_tensor_index = op->inputs()->Get(1);
            auto paddings_tensor = subgraph->tensors()->Get(paddings_tensor_index);

            
            auto paddings_buffer = tf_model->buffers()->Get(paddings_tensor->buffer())->data();
            const int32_t* paddings_data = reinterpret_cast<const int32_t*>(paddings_buffer->data());

            std::cout << "operator Pad - Padding details:" << std::endl;
            
            auto input_tensor_index = op->inputs()->Get(0);
            auto input_tensor = subgraph->tensors()->Get(input_tensor_index);
            auto input_buffer = tf_model->buffers()->Get(input_tensor->buffer())->data();
            uint8_t* input_data = const_cast<uint8_t*>(input_buffer->data());

            
            std::cout << "left padding  : " << paddings_data[4] << std::endl;
            std::cout << "right padding  : " << paddings_data[5] << std::endl;
            std::cout << "up padding  : " << paddings_data[2] << std::endl;
            std::cout << "down padding  : " << paddings_data[3] << std::endl;

            auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
            // [-1, h, w, c]
            auto output_shape = output_tensor->shape();

            
            // uint16_t output_x = x + paddings_data[4] + paddings_data[5];
            // uint16_t output_y = y + paddings_data[2] + paddings_data[3];
            // uint16_t output_ch = ch;
            delete[] output_data_buffer;
            output_data_buffer = new float[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];


            std::cout << "Padding operator... " << std::endl;
            for (int j = 0; j < 10 ; j++)
            {
                std::cout << input_data_buffer[j] << " ";
            }
            std::cout << std::endl;

            pad2d(
                input_data_buffer, x, y, ch, 
                paddings_data, 
                output_data_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                0, i);

            std::cout << "Padding applied and stored in output_data." << std::endl;

            x = output_shape->Get(1);
            y = output_shape->Get(2);
            ch = output_shape->Get(3);

            // input_data_buffer = output_data_buffer;
            input_data_buffer = std::move(output_data_buffer);


            func_count++;
        }
        break;

        case tflite::BuiltinOperator_ADD:
        {
            std::cout << "operator : " << i << "  ADD" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;
            std::cout << "ADD Channel is : " << x * y * ch << std::endl;

            delete[] output_data_buffer;
            output_data_buffer = new float[x * y * ch];

            std::cout << "2.temp indexing .. " << std::endl;
            for(int j = 0; j < 10; j++)
            {
                std::cout << temp_add_data_buffer[j] <<  " ";
            }
            std::cout << std::endl;

            for(size_t i = 0; i < x * y * ch; ++i) {
                output_data_buffer[i] = input_data_buffer[i] + temp_add_data_buffer[i];
            }

            std::ofstream out_file(layer_path + "add_" + std::to_string(i) + ".txt");
            if (out_file.is_open()) {
                // Write output_data_buffer[i] by channel
                for (int oc = 0; oc < ch; ++oc) {
                    out_file << "Output Channel " << oc + 1 << ":" << std::endl;
                    for (int oy = 0; oy < y; ++oy) {
                        for (int ox = 0; ox < x; ++ox) {
                            int output_index = (oy * x + ox) * ch + oc;
                            out_file << static_cast<int>(output_data_buffer[output_index]) << " ";
                        }
                        out_file << std::endl;
                    }
                    out_file << std::endl;
                }
                out_file.close(); // Close the file after writing
                std::cout << "Output written to add_" << i << ".txt" << std::endl;
            } else {
                std::cerr << "Unable to open file for writing!" << std::endl;
            }

            // input_data_buffer = output_data_buffer;
            input_data_buffer = std::move(output_data_buffer);
            func_count++;
        
        }
        break;

    case tflite::BuiltinOperator_MEAN:
    {
        std::cout << "operator : " << i << "  MEAN" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        // 축 텐서 가져오기
        auto axes_tensor_index = op->inputs()->Get(1);
        auto axes_tensor = subgraph->tensors()->Get(axes_tensor_index);
        auto axes_buffer = tf_model->buffers()->Get(axes_tensor->buffer())->data();
        const int32_t* axes_data = reinterpret_cast<const int32_t*>(axes_buffer->data());
        
        std::cout << "axes 0 : " << axes_data[0]  << "axes 1 : " << axes_data[1] << std::endl;
        

        // 출력 텐서 초기화
        auto output_tensor_index = op->outputs()->Get(0);
        auto output_tensor = subgraph->tensors()->Get(output_tensor_index);

        int output_size = output_tensor->shape()->Get(1);
        std::cout << "output_size is : " << output_size << std::endl;
        
        delete[] mean_data_buffer;
        mean_data_buffer = new float[output_size];
        std::fill(mean_data_buffer, mean_data_buffer + output_size, 0.0f);
        // 평균 계산할 축 파악
        bool reduce_y = false;
        bool reduce_x = false;

        for (int i = 0; i < 2; ++i) {
            int axis = axes_data[i];
            if (axis == 1) reduce_y = true;
            if (axis == 2) reduce_x = true;
        }

        std::ofstream out_file(layer_path + "mean_" + std::to_string(i) + ".txt");

        if (!out_file) {
            std::cerr << "Error: Could not create file at path: " << layer_path << std::endl;
        }

        std::cout << "reduce_y is " << reduce_y << std::endl;
        std::cout << "reduce_x is " << reduce_x << std::endl;

        // MEAN 연산 수행
        if (reduce_y && reduce_x) {
            std::cout << "MEAN 연산 수행 해용" << std::endl;
            // x, y 모두 축소: ch에 대한 값들을 축소
            for (int k = 0; k < ch; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < y; ++i) {
                    for (int j = 0; j < x; ++j) {
                        sum += input_data_buffer[i * x * ch + j * ch + k];
                    }
                }
                mean_data_buffer[k] = sum / (x * y);
                out_file << mean_data_buffer[k] << " ";
                if(k == 0) {
                    std::cout << "4.mean data buffer is : " << mean_data_buffer[0] << std::endl;
                }
            }
            std::cout << "1.mean data buffer is : " << mean_data_buffer[0] << std::endl;
        }    
        out_file << std::endl;
        out_file.close();

        std::cout << "2.mean data buffer is : " << mean_data_buffer[0] << std::endl;
        std::cout << "MEAN operation applied and stored in output_data." << std::endl;

        y = output_tensor->shape()->Get(0);
        std::cout << "y is : " <<  y  << std::endl;

        func_count++;
    }
    break;

    case tflite::BuiltinOperator_FULLY_CONNECTED:
    {
        std::cout << "operator : " << i << "  Fully Connected" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        // Get the input tensor (mean_data_buffer is used as the input data)
        std::cout << "3.mean data buffer is : " << mean_data_buffer[0] << std::endl;

        // Get the weights (kernel) tensor
        auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
        auto filter_shape = filter_tensor->shape();
        auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer());
        const float* filter_data = reinterpret_cast<const float*>(filter_buffer->data()->data());
        size_t filter_buffer_size = filter_buffer->data()->size() / sizeof(float);
        std::vector<float> flat_filter_buffer(filter_buffer_size);
        
        std::copy(filter_data, filter_data + filter_buffer_size, flat_filter_buffer.begin());
        std::cout << "flat filter buffer first index " <<  i <<  " is  : "  << flat_filter_buffer[0] << std::endl;
        
        // Get the output tensor and its size
        auto output_tensor_index = op->outputs()->Get(0);
        auto output_tensor = subgraph->tensors()->Get(output_tensor_index);
        int output_size = output_tensor->shape()->Get(1);

        std::cout << "fully connected output size is : " << output_size << std::endl;

        
        // Input size
        int input_size = 1280;  // Number of input nodes (1280 in your case)
        std::cout << "input size is : " << input_size << std::endl;
        std::cout << "output_nodes is : " << output_size << std::endl;

        std::ofstream out_file(layer_path + "fully_connected_" + std::to_string(i) + ".txt");

        // Perform the Fully Connected operation
        for (int i = 0; i < output_size; ++i) {
            float sum = 0;
            for (int j = 0; j < input_size; ++j) {
                sum += mean_data_buffer[j] * flat_filter_buffer[i * input_size + j];
            }

            // Apply ReLU activation function
            // sum = std::max(0.0f, sum);
            temp_sum = sum;
            // Write to file
            out_file << sum << " ";

        }
        out_file.close();
        delete[] mean_data_buffer;
        mean_data_buffer = nullptr;
        
        std::cout << "Fully connected operation applied and stored in output_data_buffer." << std::endl;

        // Optionally, store x, y, ch based on the output shape if needed
        x = 1;  // Fully connected layer output is usually 1-dimensional
        y = 1;
        ch = output_size;

        func_count++;
    }
    break;

    
    case tflite::BuiltinOperator_LOGISTIC:
    {
        std::cout << "operator : " << i << "  Logistic" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        // Apply the logistic function to all elements in fully_connected_data_buffer
        int output_size = ch;  // Assuming ch is the size of the fully connected output

        temp_sum = 1.0f / (1.0f + std::exp(-temp_sum));
        

        // Optionally, store the final prediction
        if (temp_sum > 0.5) {
            std::cout << "Prediction: Success with probability " << temp_sum << std::endl;
        } else {
            std::cout << "Prediction: Failure with probability " << temp_sum << std::endl;
        }

        std::cout << "Logistic function applied and stored in output_data_buffer." << std::endl;

        func_count++;
    }
    break;

    default:
        break;
    }
    }

    std::cout << "func count is : " << func_count <<  std::endl;

}

void CodeGenerator::clearBuffer() {
    std::cout << "소멸자 호출" << std::endl;
    if (imageBuffer != nullptr) {
        delete[] imageBuffer;
        imageBuffer = nullptr;
    }
}

void print_padded_channel_to_file(std::ofstream &out_file, float *data, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            out_file << static_cast<int>(data[i * width + j]) << " ";
        }
        out_file << std::endl;
    }
}

void CodeGenerator::conv2d(
    float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const float *kernel, size_t kernelSize, int kernel_height, int kernel_width,
    const float *bias, 
    const float output_activation_min,
    const float output_activation_max, 
    float *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    uint8_t stride_value, int layer_index, bool relu_flag, bool pad_is_same
)
{
    int pad_h = 0, pad_w = 0;
    if (pad_is_same) {
        pad_h = (kernel_height - 1) / 2;
        pad_w = (kernel_width - 1) / 2;
    }

    // If padding is 'same', create a padded input tensor
    int padded_x = input_x + 2 * pad_w;
    int padded_y = input_y + 2 * pad_h;
    std::vector<float> padded_input(padded_x * padded_y * input_ch, 0.0f);

    

    if (pad_is_same) {
        for (int c = 0; c < input_ch; ++c) {
            for (int y = 0; y < input_y; ++y) {
                for (int x = 0; x < input_x; ++x) {
                    int input_index = (y * input_x + x) * input_ch + c;
                    int padded_index = ((y + pad_h) * padded_x + (x + pad_w)) * input_ch + c;
                    padded_input[padded_index] = input[input_index];
                }
            }
        }
    } else {
        // No padding, use original input
        padded_input.assign(input, input + input_x * input_y * input_ch);
    }


    // 여기까지 로직 이상 없음

    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    std::cout << "kerner index data is : " << kernel[4] << std::endl;
    
    int batch_size = 1;

    std::cout << "kernel width : " << kernel_width << std::endl;
    std::cout << "kernel height : " << kernel_height << std::endl;
    std::cout << "kernel channel : " << static_cast<int>(input_ch) << std::endl;
    std::cout << "ouptut channel : " << output_ch << std::endl;
    std::cout << "kernel size : " << kernelSize << std::endl;

    int flag = 0;

    int kernel_size = kernel_width * kernel_height * input_ch; 

    for (int n = 0; n < batch_size; ++n) { // Iterate over batch
        for (int oc = 0; oc < output_channels; ++oc) { // Iterate over the filters (output channels)
            for (int oy = 0; oy < output_height; ++oy) { // Iterate over the output spatial dimensions
                for (int ox = 0; ox < output_width; ++ox) {
                    float acc = 0.0f;

                    // Perform convolution
                    for (int ic = 0; ic < input_ch; ++ic) { // Iterate over input channels
                        for (int ky = 0; ky < kernel_height; ++ky) {
                            for (int kx = 0; kx < kernel_width; ++kx) {
                                int ix = ox * stride_value + kx;
                                int iy = oy * stride_value + ky;

                                if (ix >= 0 && ix < padded_x && iy >= 0 && iy < padded_y) {
                                    int input_index = (n * padded_y + iy) * padded_x * input_ch + ix * input_ch + ic;
                                    int flat_kernel_index = oc * kernel_size + ky * kernel_width * input_ch + kx * input_ch + ic;
                                    acc += padded_input[input_index] * kernel[flat_kernel_index];
                                }
                            }
                        }
                    }

                    // Add the bias (if used)
                    acc += bias[oc];

                    // Apply ReLU or ReLU6 activation if specified
                    if (relu_flag) {
                        acc = std::max(output_activation_min, std::min(acc, output_activation_max));
                    }

                    // Store the result in the output tensor
                    int output_index = ((n * output_height + oy) * output_width + ox) * output_channels + oc;
                    output[output_index] = acc;
                }
            }
        }
    }

    std::ofstream out_file(layer_path + "outputConv2d" + std::to_string(layer_index) + ".txt");
    if (out_file.is_open()) {
        for (int oc = 0; oc < output_ch; ++oc) {
            out_file << "Output Channel " << oc + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_y; ++oy) {
                for (int ox = 0; ox < output_x; ++ox) {
                    int output_index = (oy * output_x + ox) * output_ch + oc;
                    out_file <<  static_cast<int>(output[output_index]) << " ";
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

void CodeGenerator::depthwiseConv2d(
    float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const float *kernel, const float *bias, size_t kernelSize, int kernel_height, int kernel_width,
    const float output_activation_min,
    const float output_activation_max, 
    float *output, const uint16_t output_x, const uint16_t output_y, const uint16_t output_ch,
    uint8_t stride_value, int layer_index, bool relu_flag, bool pad_is_same
)
{
    int pad_h = 0, pad_w = 0;
    if (pad_is_same) {
        pad_h = (kernel_height - 1) / 2;
        pad_w = (kernel_width - 1) / 2;
    }

    // If padding is 'same', create a padded input tensor
    int padded_x = input_x + 2 * pad_w;
    int padded_y = input_y + 2 * pad_h;
    std::vector<float> padded_input(padded_x * padded_y * input_ch, 0.0f);

    

    if (pad_is_same) {
        for (int c = 0; c < input_ch; ++c) {
            for (int y = 0; y < input_y; ++y) {
                for (int x = 0; x < input_x; ++x) {
                    int input_index = (y * input_x + x) * input_ch + c;
                    int padded_index = ((y + pad_h) * padded_x + (x + pad_w)) * input_ch + c;
                    padded_input[padded_index] = input[input_index];
                }
            }
        }
    } else {
        // No padding, use original input
        padded_input.assign(input, input + input_x * input_y * input_ch);
    }

    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    int batch_size = 1;
    int kernel_size = kernel_width * kernel_height * input_ch; 

    // Depthwise convolution operation
    for (int n = 0; n < batch_size; ++n) { // Iterate over batch
        for (int oc = 0; oc < output_channels; ++oc) { // Iterate over the filters (output channels)
            for (int oy = 0; oy < output_height; ++oy) { // Iterate over the output spatial dimensions
                for (int ox = 0; ox < output_width; ++ox) {
                    float acc = 0.0f;

                    // Perform depthwise convolution (each output channel corresponds to an input channel)
                    for (int ky = 0; ky < kernel_height; ++ky) {
                        for (int kx = 0; kx < kernel_width; ++kx) {
                            int ix = ox * stride_value + kx;
                            int iy = oy * stride_value + ky;

                            if (ix >= 0 && ix < padded_x && iy >= 0 && iy < padded_y) {
                                // Since it's depthwise, input channel = output channel (oc)
                                int input_index = (n * padded_y + iy) * padded_x * input_ch + ix * input_ch + oc;
                                int flat_kernel_index = oc * kernel_size + ky * kernel_width + kx;
                                acc += padded_input[input_index] * kernel[flat_kernel_index];
                            }
                        }
                    }

                    // Add the bias (if used)
                    acc += bias[oc];

                    // Apply ReLU or ReLU6 activation if specified
                    if (relu_flag) {
                        acc = std::max(output_activation_min, std::min(acc, output_activation_max));
                    }

                    // Store the result in the output tensor
                    int output_index = ((n * output_height + oy) * output_width + ox) * output_channels + oc;
                    output[output_index] = acc;
                }
            }
        }
    }

    std::cout << "depthwise output indexing..." << std::endl;
    for (int j = 0; j < 10; j++) {
        std::cout << output[j] << " ";
    }
    std::cout << std::endl;

    // Write output tensor to file
    std::ofstream out_file(layer_path + "depthWiseConv2d_" + std::to_string(layer_index) + ".txt");
    if (out_file.is_open()) {
        for (int ic = 0; ic < input_ch; ++ic) {
            out_file << "Output Channel " << ic + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_height; ++oy) {
                for (int ox = 0; ox < output_width; ++ox) {
                    int output_index = (oy * output_width + ox) * output_channels + ic;
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



void CodeGenerator::pad2d(
    float *input, const uint8_t input_x, const uint8_t input_y, const uint8_t input_ch,
    const int32_t* paddings,  // 패딩 정보를 받는 파라미터
    float *output,  uint16_t output_x, uint16_t output_y, uint16_t output_ch, // 결과를 저장할 output 포인터
    int8_t pad_value, int layer_index
)
{
    const int output_width = output_x;
    const int output_height = output_y;
    const int output_channels = output_ch;

    std::cout << "왼쪽 패딩 : " << paddings[4] << " 오른쪽 패딩 : " << paddings[5] << std::endl;
    std::cout << "위쪽 패딩 : " << paddings[2] << " 아래쪽 패딩 : " << paddings[3] << std::endl;

    // 데이터 복사 포인터 설정
    float *input_ptr = input;

    // 입력 데이터를 패딩된 위치에 직접 복사
    for (int c = 0; c < input_ch; ++c) {
        for (int oy = 0; oy < output_height; ++oy) {
            for (int ox = 0; ox < output_width; ++ox) {
                int output_index = (oy * output_width + ox) * output_channels + c;
                
                if (oy < paddings[2] || oy >= (output_height - paddings[3]) ||
                    ox < paddings[4] || ox >= (output_width - paddings[5])) {
                    // 패딩 영역
                    output[output_index] = static_cast<int>(pad_value);
                } else {
                    // 원본 데이터 영역
                    int input_index = ((oy - paddings[2]) * input_x + (ox - paddings[4])) * input_ch + c;
                    output[output_index] = input[input_index];
                }
            }
        }
    }

    std::cout << "padding x size is : " << output_x <<  std::endl;
    std::cout << "padding y size is : " << output_y <<  std::endl;


    std::cout << "padding ouptut indexing..." << std::endl;
    for(int j = 0; j < 10; j++)
    {
        std::cout << output[j] << " ";
    }
    std::cout << std::endl;

    // 패딩된 결과를 파일에 저장
    std::ofstream out_file(layer_path + "pad_" + std::to_string(layer_index) + ".txt");
    if (out_file.is_open()) {
        for (int oc = 0; oc < output_ch; ++oc) {
            out_file << "Output Channel " << oc + 1 << ":" << std::endl;
            for (int oy = 0; oy < output_height; ++oy) {
                for (int ox = 0; ox < output_width; ++ox) {
                    int output_index = (oy * output_width + ox) * output_channels + oc;
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

