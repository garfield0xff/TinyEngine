#include "profiler.h"
#include <iostream>
#include <fstream>
#include <json/json.h>

/* 
    Save Model Information
    (Number of Tenser)  
    (Number of operator) 
    (Input Tensor shape) 
    (operator type)   
    (Tensor Type)
    (Tensor Shape) 
    (Tensor Buffer Size) 
*/

void printTensorInfo(const tflite::Tensor *tensor, const tflite::Model *model, std::ofstream& output_file) {
    
    output_file << "Tensor Name: " << tensor->name()->str() << std::endl;
    output_file << "Tensor Type: " << tflite::EnumNameTensorType(tensor->type()) << std::endl;

    auto shape = tensor->shape();
    output_file << "Tensor Shape: [";
    for (size_t i = 0; i < shape->size(); ++i) {
        output_file << shape->Get(i);
        if (i < shape->size() - 1) {
            output_file << ", ";
        }
    }

    output_file << "]" << std::endl;

    auto buffer_index = tensor->buffer();
    auto buffer = model->buffers()->Get(buffer_index);
    if (buffer->data()) {
        output_file << "Tensor Buffer Size: " << buffer->data()->size() << " bytes" << std::endl;
    } else {
        output_file << "Tensor Buffer Size: 0 bytes" << std::endl;
    }

    output_file << "---------------------------------------------------" << "\n";
}


void TensorProfiler::saveModelInfo(const tflite::Model* tf_model, const string& output_path) 
{
    
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open file" << "\n";
        return;
    }

    if(tf_model->subgraphs()->size() == 0) {
        std::cerr << "Model has no subgrpahs" << "\n";
        return;
    }

    const tflite::SubGraph* subgraph = tf_model->subgraphs()->Get(0);
    output_file << "Number of Tensors: " << subgraph->tensors()->size() << "\n";

    // all operator size 
    output_file << "Number of Operators: " << subgraph->operators()->size() << "\n";

    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
        auto op = subgraph->operators()->Get(i);
        auto opcode_index = op->opcode_index();
        auto opcode = tf_model->operator_codes()->Get(opcode_index)->builtin_code();
        // std::cout << "Operator : " << tflite::EnumNameBuiltinOperator(opcode) << "\n";
        output_file << "Operator : " << tflite::EnumNameBuiltinOperator(opcode) << "\n";
        output_file << "----------------- Input Tensors -------------------" << "\n";
        for (size_t i = 0; i < op->inputs()->size(); ++i) {
            auto input_tensor_index = op->inputs()->Get(i);

            if (input_tensor_index != -1) {
                auto input_tensor = subgraph->tensors()->Get(input_tensor_index);
                printTensorInfo(input_tensor, tf_model, output_file);
            }
        }
        output_file << "---------------------------------------------------" << "\n";
        output_file << "----------------- Output Tensors -------------------" << "\n";
        for (size_t i = 0; i < op->outputs()->size(); ++i) {
            auto output_tensor_index = op->outputs()->Get(i);
                if (output_tensor_index != -1) {
                    auto output_tensor = subgraph->tensors()->Get(output_tensor_index);
                    printTensorInfo(output_tensor, tf_model, output_file);
                }
        }
        output_file << "---------------------------------------------------" << "\n";
        output_file << "\n";
        output_file << "\n";
        output_file << "\n";

    }
}

/*
    224 x 224 image interference 
    this model designed for predict person
*/

void TensorProfiler::runInference(tflite::Interpreter *interpreter, const cv::Mat& input_img){


    if (!interpreter) {
        std::cerr << "Failed to build interpreter!" << "\n";
        return;
    }

    // allocate tensor
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << "\n";
        return;
    }

    //resize image to model input size
    cv::Mat preprocessed_img;
    cv::resize(input_img, preprocessed_img, cv::Size(224, 224));

    // Mat -> float32[0, 1]
    preprocessed_img.convertTo(preprocessed_img, CV_32FC3, 1.0 / 255.0);

    // copy data
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor, input_img.data, input_img.total() * input_img.elemSize());

    // run Model
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter!" << "\n";
        return;
    }

    // get result from output tensor
    float* output_tensor = interpreter->typed_output_tensor<float>(0);

    // check result tensor
    if (output_tensor[0] > 0.7) 
    {
        std::cout << "Prediction : This image contains a person." << "\n"; 
    } else {
        std::cout << "Prediction : This image does not contain a person." << "\n"; 
    }
}

    





