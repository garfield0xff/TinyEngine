// code_generator.cpp
#include "code_generator.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <validator.h>

using namespace cv;

const std::string layer_path = "/Users/gyujinkim/Desktop/Github/TinyEngine/code_generator/test/layer/";

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

        out_file.close(); 
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }

    for(int i = 0; i < 10; i ++)
    {
        std::cout << "flat imgae buffer is :" << _imageBuffer[i] << " ";
    }
    std::cout << std::endl;
}

void CodeGenerator::parseTFModel(const tflite::Model *tf_model)
{
    if(tf_model->subgraphs()->size() == 0)
    {
        std::cerr << "Model has no graph" << "\n";
        return;
    }

    const tflite::SubGraph* subgraph = tf_model->subgraphs()->Get(0);

    int16_t x = input_x;
    int16_t y = input_y;
    int16_t ch = input_ch;
    float* input_data_buffer = imageBuffer; 
    float* output_data_buffer = nullptr; 
    float* temp_add_data_buffer = nullptr;
    float* mean_data_buffer = nullptr;
    float temp_sum = 0;

    bool flag = false;
    int func_count = 0;

    Validator v1;

    v1.set_layer_folder_path(layer_path);

    for(size_t i = 0; !flag && i < subgraph->operators()->size(); ++i)
    {

        auto op = subgraph->operators()->Get(i);
        auto opcode_index = op->opcode_index();
        auto opcode = tf_model->operator_codes()->Get(opcode_index)->builtin_code();

        switch (opcode)
        {
        case tflite::BuiltinOperator_CONV_2D:
        {

            std::cout << "operator : " << i << " Convolution" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;

            const tflite::Conv2DOptions* conv_options = op->builtin_options_as_Conv2DOptions();
            tflite::ActivationFunctionType activation_type = conv_options->fused_activation_function();
            uint8_t stride = conv_options-> stride_h();

            bool relu_flag = false;
            bool pad_is_valid = true;
            
            if (activation_type == tflite::ActivationFunctionType::ActivationFunctionType_RELU6) relu_flag = true;

            if (conv_options->padding() == tflite::Padding::Padding_VALID)pad_is_valid = false;
            else if (conv_options->padding() == tflite::Padding::Padding_SAME)pad_is_valid = true;
        
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
            
            auto bias_tensor = subgraph->tensors()->Get(op->inputs()->Get(2));
            auto bias_shape = bias_tensor->shape();
            auto bias_type = bias_tensor->type();
            auto bias_buffer = tf_model->buffers()->Get(bias_tensor->buffer());
            const float* bias_data = reinterpret_cast<const float*>(bias_buffer->data()->data());
            size_t bias_buffer_size = bias_buffer->data()->size() / sizeof(float);
            std::vector<float> flat_bias_buffer(bias_buffer_size);
            
            
            std::copy(filter_data, filter_data + filter_buffer_size, flat_filter_buffer.begin());
            std::copy(bias_data, bias_data + bias_buffer_size, flat_bias_buffer.begin());
        
            auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
            auto output_shape = output_tensor->shape();

            delete[] output_data_buffer;
            output_data_buffer = new float[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];
            std::cout << "output buffer size is : "<< output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3) << std::endl;

            v1.conv2d(
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
            } else if (activation_type == tflite::ActivationFunctionType::ActivationFunctionType_RELU6) {
                std::cout << "ReLU6 activation is included in this convolution." << std::endl;
                relu_flag = true;
            } else {
                std::cout << "No ReLU or ReLU6 activation is included in this convolution." << std::endl;
            }

            if (conv_options->padding() == tflite::Padding::Padding_VALID)pad_is_valid = false;
            
            auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
            auto filter_shape = filter_tensor->shape();
            auto filter_type = filter_tensor->type();
            auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer());
            const float* filter_data = reinterpret_cast<const float*>(filter_buffer->data()->data());
            size_t filter_buffer_size = filter_buffer->data()->size() / sizeof(float);
            std::vector<float> flat_filter_buffer(filter_buffer_size);

            
            auto bias_tensor = subgraph->tensors()->Get(op->inputs()->Get(2));
            auto bias_shape = bias_tensor->shape();
            auto bias_type = bias_tensor->type();
            auto bias_buffer = tf_model->buffers()->Get(bias_tensor->buffer());
            const float* bias_data = reinterpret_cast<const float*>(bias_buffer->data()->data());
            size_t bias_buffer_size = bias_buffer->data()->size() / sizeof(float);
            std::vector<float> flat_bias_buffer(bias_buffer_size);
            
            std::copy(filter_data, filter_data + filter_buffer_size, flat_filter_buffer.begin());
            std::copy(bias_data, bias_data + bias_buffer_size, flat_bias_buffer.begin());
            
            auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
            auto output_shape = output_tensor->shape();
        
            int filter_height = filter_shape->Get(1);
            int filter_width = filter_shape->Get(2);

            delete[] output_data_buffer;
            output_data_buffer = new float[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];
            
            v1.depthwiseConv2d(
                input_data_buffer, x, y, ch,
                flat_filter_buffer.data(),  filter_buffer_size, filter_height, filter_width, 
                flat_bias_buffer.data(),
                0.0, 6.0,
                output_data_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                stride, i, relu_flag, pad_is_valid
            );

            x = output_shape->Get(1);
            y = output_shape->Get(2);
            ch = output_shape->Get(3);

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

            auto input_tensor_index = op->inputs()->Get(0);
            auto input_tensor = subgraph->tensors()->Get(input_tensor_index);
            auto input_buffer = tf_model->buffers()->Get(input_tensor->buffer())->data();
            uint8_t* input_data = const_cast<uint8_t*>(input_buffer->data());

            auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
            auto output_shape = output_tensor->shape();

            delete[] output_data_buffer;
            output_data_buffer = new float[output_shape->Get(1) * output_shape->Get(2) * output_shape->Get(3)];

            v1.pad2d(
                input_data_buffer, x, y, ch, 
                paddings_data, 
                output_data_buffer, static_cast<int16_t>(output_shape->Get(1)), static_cast<int16_t>(output_shape->Get(2)), static_cast<int16_t>(output_shape->Get(3)),
                0, i);

            x = output_shape->Get(1);
            y = output_shape->Get(2);
            ch = output_shape->Get(3);

            input_data_buffer = std::move(output_data_buffer);
            func_count++;
        }
        break;

        case tflite::BuiltinOperator_ADD:
        {
            std::cout << "operator : " << i << "  ADD" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;

            delete[] output_data_buffer;
            output_data_buffer = new float[x * y * ch];

            v1.add(
                input_data_buffer, temp_add_data_buffer, x, y, ch, output_data_buffer, i
            );

            input_data_buffer = std::move(output_data_buffer);
            func_count++;
        }
        break;

        case tflite::BuiltinOperator_MEAN:
        {
            std::cout << "operator : " << i << "  MEAN" << std::endl;
            std::cout << "------------------------------------------------" << std::endl;

            auto axes_tensor_index = op->inputs()->Get(1);
            auto axes_tensor = subgraph->tensors()->Get(axes_tensor_index);
            auto axes_buffer = tf_model->buffers()->Get(axes_tensor->buffer())->data();
            const int32_t* axes_data = reinterpret_cast<const int32_t*>(axes_buffer->data());
            
            std::cout << "axes 0 : " << axes_data[0]  << "axes 1 : " << axes_data[1] << std::endl;
        
            auto output_tensor_index = op->outputs()->Get(0);
            auto output_tensor = subgraph->tensors()->Get(output_tensor_index);

            int output_size = output_tensor->shape()->Get(1);
            std::cout << "output_size is : " << output_size << std::endl;
            
            delete[] mean_data_buffer;
            mean_data_buffer = new float[output_size];
            std::fill(mean_data_buffer, mean_data_buffer + output_size, 0.0f);
            
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

            if (reduce_y && reduce_x) {
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
            func_count++;
        }
        break;

    case tflite::BuiltinOperator_FULLY_CONNECTED:
    {
        std::cout << "operator : " << i << "  Fully Connected" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
        auto filter_shape = filter_tensor->shape();
        auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer());
        const float* filter_data = reinterpret_cast<const float*>(filter_buffer->data()->data());
        size_t filter_buffer_size = filter_buffer->data()->size() / sizeof(float);
        std::vector<float> flat_filter_buffer(filter_buffer_size);        

        std::copy(filter_data, filter_data + filter_buffer_size, flat_filter_buffer.begin());
        
        
        auto output_tensor_index = op->outputs()->Get(0);
        auto output_tensor = subgraph->tensors()->Get(output_tensor_index);
        int output_size = output_tensor->shape()->Get(1);


        int input_size = 1280;  
        std::cout << "input size is : " << input_size << std::endl;
        std::cout << "output_nodes is : " << output_size << std::endl;

        std::ofstream out_file(layer_path + "fully_connected_" + std::to_string(i) + ".txt");

        
        for (int i = 0; i < output_size; ++i) {
            float sum = 0;
            for (int j = 0; j < input_size; ++j) {
                sum += mean_data_buffer[j] * flat_filter_buffer[i * input_size + j];
            }
            temp_sum = sum;
            out_file << sum << " ";

        }
        out_file.close();
        delete[] mean_data_buffer;
        mean_data_buffer = nullptr;
        
        x = 1;  
        y = 1;
        ch = output_size;
        func_count++;
    }
    break;

    
    case tflite::BuiltinOperator_LOGISTIC:
    {
        std::cout << "operator : " << i << "  Logistic" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        int output_size = ch;  

        temp_sum = 1.0f / (1.0f + std::exp(-temp_sum));
        
        if (temp_sum > 0.5) std::cout << "Prediction: Success with probability " << temp_sum << std::endl;
        else std::cout << "Prediction: Failure with probability " << temp_sum << std::endl;
    
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



