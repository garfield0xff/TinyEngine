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

        switch (opcode)
        {
        case tflite::BuiltinOperator_CONV_2D:
            {
                const tflite::Conv2DOptions* conv_options = op->builtin_options_as_Conv2DOptions();
                
                // Stride 정보 추출
                int stride_h = conv_options->stride_h();
                int stride_w = conv_options->stride_w();
                std::cout << "Stride: (" << stride_h << ", " << stride_w << ")" << std::endl;

                // Input 텐서 정보 추출
                auto input_tensor = subgraph->tensors()->Get(op->inputs()->Get(0));
                auto input_shape = input_tensor->shape();
                std::cout << "Input shape: ";
                for (int j = 0; j < input_shape->size(); j++)
                    std::cout << input_shape->Get(j) << " ";
                std::cout << std::endl;

                auto input_type = input_tensor->type();
                std::cout << "Input type: " << tflite::EnumNameTensorType(input_type) << std::endl;

                // Filter (weight) 텐서 정보 추출
                auto filter_tensor = subgraph->tensors()->Get(op->inputs()->Get(1));
                auto filter_shape = filter_tensor->shape();
                std::cout << "Filter shape: ";
                for (int j = 0; j < filter_shape->size(); j++)
                    std::cout << filter_shape->Get(j) << " ";
                std::cout << std::endl;

                auto filter_type = filter_tensor->type();
                std::cout << "Filter type: " << tflite::EnumNameTensorType(filter_type) << std::endl;

                // Min/Max 값 추출
                float min = INT8_MAX;
                float max = 0.;

                auto filter_buffer = tf_model->buffers()->Get(filter_tensor->buffer())->data();
                if (filter_buffer && filter_buffer->size() > 0) {
                    const float* filter_data = reinterpret_cast<const float*>(filter_buffer->data());
                    size_t filter_size = 1;
                    for (int j = 0; j < filter_shape->size(); ++j) {
                        filter_size *= filter_shape->Get(j);
                    }

                    std::cout << "Filter values: ";
                    for (size_t j = 0; j < filter_size; ++j) {
                        float temp = static_cast<float>(filter_data[j]);
                        if(min > temp)min = temp;
                        if(max < temp)max = temp;
                        
                        std::cout << temp << " ";
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "Filter tensor has no data." << std::endl;
                }
                std::cout << "min is  : " << min << " ";
                std::cout << "max is  : " << max << std::endl;
                int8_t test = bit8_quantization(-0.304157, min, max);
                std::cout << "quantization value is : "  <<  static_cast<int>(test) << std::endl;

                // bias 텐서 정보 추출 (존재할 경우)
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
                    if (bias_buffer && bias_buffer->size() > 0) {
                        const float* bias_data = reinterpret_cast<const float*>(bias_buffer->data());
                        size_t bias_size = bias_shape->size() > 0 ? bias_shape->Get(0) : 0;

                        std::cout << "Bias values: ";
                        for (size_t j = 0; j < bias_size; ++j) {
                            std::cout << bias_data[j] << " ";
                        }
                        std::cout << std::endl;
                    } else {
                        std::cout << "Bias tensor has no data." << std::endl;
                    }
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