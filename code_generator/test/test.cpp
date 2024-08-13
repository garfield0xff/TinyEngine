#include <iostream>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>

int8_t bit8_quantization(float value, float min, float max) {
    float scale = (max - min) / 255.0;
    int quantized_value = round((value - min) / scale);

    if(quantized_value < -128) quantized_value = - 128;
    if(quantized_value > 127) quantized_value = 127;

    return static_cast<int8_t>(quantized_value);
};

void conv2d_kernel3x3 (
    int8_t *input, const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
    int8_t *kernel, const int32_t *bias,
    const float* scales,
    const int32_t output_offset, const int32_t input_offset,
    const int32_t output_activation_min, const int32_t output_activation_max,
    const uint16_t output_x, const uint16_t output_y,
    const uint16_t output_ch, const uint8_t stride, int16_t *runtime_buf, int8_t pad_value
) 
{
    int c, i, j;
    int16_t *cols_8b_start = (int16_t *)runtime_buf;
    int16_t *cols_8b = (int16_t *)cols_8b_start;

    int16_t PAD8 = pad_value;

    for(i = 0; i < input_x + 2; i++) {
        *cols_8b++ = PAD8;
    }

    for(i = 0; i < input_y; i++)
    {
        *cols_8b++ = PAD8;
        cols_8b += input_x;
        *cols_8b++ = PAD8;
    }

    for(i = 0; i < input_x; i++)
    {
        *cols_8b++ = PAD8;
    }

}








void paseModel(const tflite::Model* tf_model)
{
    if(tf_model->subgraphs()->size() == 0)
    {
        std::cerr << "Model has no graph" << "\n";
        return;
    }

    const tflite::SubGraph* subgraph = tf_model->subgraphs()->Get(0);
    
    bool flag = false;

    for(size_t i = 0; i < subgraph->operators()->size(); ++i)
    {
        if(flag)break;
        auto op = subgraph->operators()->Get(i);
        auto opcode_index = op->opcode_index();
        auto opcode = tf_model->operator_codes()->Get(opcode_index)->builtin_code();
        
        std::vector<int8_t> output;

        std::cout << "vector initial size is : " << output.size() << std::endl;

        switch (opcode)
        {
        case tflite::BuiltinOperator_CONV_2D:
            {
                const tflite::Conv2DOptions* conv_options = op->builtin_options_as_Conv2DOptions();
                
                int stride_h = conv_options->stride_h();

                // Input Tensor Info 
                // input, input_x, input_y, input_ch, input_type 
                auto input_tensor = subgraph->tensors()->Get(op->inputs()->Get(0));
                auto input_shape = input_tensor->shape();
                std::cout << "Input shape: " << input_shape;
                std::cout << std::endl;

                auto input_type = input_tensor->type();
                std::cout << "Input type: " << tflite::EnumNameTensorType(input_type) << std::endl;

                // Kernel Tensor Info
                // kernel_shape, kernel buffer, 
                // min max to get quantization scale 
                auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
                // shape : [batch size, w, h, c]
                auto filter_shape = filter_tensor->shape();
                std::cout << "filter shape : " << filter_shape->Get(0) << " " << filter_shape->Get(1) << " " << filter_shape->Get(2) << " " << filter_shape->Get(3);
                
                std::cout << std::endl;

                auto filter_type = filter_tensor->type();
                std::cout << "Filter type: " << tflite::EnumNameTensorType(filter_type) << std::endl;

                // get MIN / MAX
                float min = INT8_MAX;
                float max = 0.;

                auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer())->data();
                if (filter_buffer && filter_buffer->size() > 0) {
                    const float* filter_data = reinterpret_cast<const float*>(filter_buffer->data());
                    size_t filter_size = 1;
                    for (int j = 0; j < filter_shape->size(); ++j) {
                        filter_size *= filter_shape->Get(j);
                        std::cout <<  "filter_shape->Get(j) = " << filter_shape->Get(j) << " ";
                    }
                    std::cout << "\n\n";

                    std::cout << "Filter values: ";
                    for (size_t j = 0; j < filter_size; ++j) {
                        float temp = static_cast<float>(filter_data[j]);
                        if(min > temp)min = temp;
                        if(max < temp)max = temp;
                        std::cout << temp << " ";
                    }
                    std::cout << "\n\n";

                } else {
                    std::cout << "Filter tensor has no data." << std::endl;
                }

                std::cout << "min is  : " << min << " ";
                std::cout << "max is  : " << max << std::endl;
                

                // Bias Tensor Info
                if (op->inputs()->size() > 2) {
                    auto bias_tensor = subgraph->tensors()->Get(op->inputs()->Get(2));
                    auto bias_shape = bias_tensor->shape();
                    std::cout << "Bias shape: ";
                    for (int j = 0; j < bias_shape->size(); j++)
                        std::cout << bias_shape->Get(j) << " ";
                    std::cout << std::endl;

                    auto bias_type = bias_tensor->type();
                    std::cout << "Bias type: " << tflite::EnumNameTensorType(bias_type) << std::endl;

                    auto bias_buffer = tf_model->buffers()->Get(bias_tensor->buffer())->data();
                    // if (bias_buffer && bias_buffer->size() > 0) {
                    //     const float* bias_data = reinterpret_cast<const float*>(bias_buffer->data());
                    //     size_t bias_size = bias_shape->size() > 0 ? bias_shape->Get(0) : 0;

                    //     std::cout << "Bias values: ";
                    //     for (size_t j = 0; j < bias_size; ++j) {
                    //         std::cout << bias_data[j] << " ";
                    //     }
                    //     std::cout << std::endl;
                    // } else {
                    //     std::cout << "Bias tensor has no data." << std::endl;
                    // }
                }

                // Output 텐서 정보 추출
                auto output_tensor = subgraph->tensors()->Get(op->outputs()->Get(0));
                auto output_shape = output_tensor->shape();
                std::cout << "Output shape: ";
                for (int j = 0; j < output_shape->size(); j++)
                    std::cout << output_shape->Get(j) << " ";
                std::cout << std::endl;

                auto output_type = output_tensor->type();
                std::cout << "Output type: " << tflite::EnumNameTensorType(output_type) << std::endl;
                flag = true;
            }
            break;
        
        default:
            break;
        }
    }

}

int main()
{
    const char* model_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/person_detection_model.tflite";



    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    const tflite::Model* tf_model = model->GetModel();

    paseModel(tf_model);

    return 0;
}