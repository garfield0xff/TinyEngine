#include "profiler.h"
#include <iostream>


void TensorProfiler::printTensorInfo(const Tensor *tensor, const Model *model) {
    std::cout << "Tensor Name: " << tensor->name()->str() << std::endl;
    std::cout << "Tensor Type: " << tflite::EnumNameTensorType(tensor->type()) << std::endl;

    auto shape = tensor->shape();
    std::cout << "Tensor Shape: [";
    for (size_t i = 0; i < shape->size(); ++i) {
        std::cout << shape->Get(i);
        if (i < shape->size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    auto buffer_index = tensor->buffer();
    auto buffer = model->buffers()->Get(buffer_index);
    if (buffer->data()) {
        std::cout << "Tensor Buffer Size: " << buffer->data()->size() << " bytes" << std::endl;
    } else {
        std::cout << "Tensor Buffer Size: 0 bytes" << std::endl;
    }
    std::cout << "----------------------" << std::endl;
}

void TensorProfiler::printOperatorInfo(const tflite::Operator *op, const tflite::Model *model, const tflite::SubGraph *subgraph) {
    auto opcode_index = op->opcode_index();
    auto opcode = model->operator_codes()->Get(opcode_index)->builtin_code();
    std::cout << "Operator: " << tflite::EnumNameBuiltinOperator(opcode) << std::endl;

    // Input Tensors
    std::cout << "Input Tensors: " << std::endl;
    for (size_t i = 0; i < op->inputs()->size(); ++i) {
        auto input_tensor_index = op->inputs()->Get(i);
        if (input_tensor_index != -1) {
            auto input_tensor = subgraph->tensors()->Get(input_tensor_index);
            printTensorInfo(input_tensor, model);
        }
    }

    // Output Tensors
    std::cout << "Output Tensors: " << std::endl;
    for (size_t i = 0; i < op->outputs()->size(); ++i) {
        auto output_tensor_index = op->outputs()->Get(i);
        if (output_tensor_index != -1) {
            auto output_tensor = subgraph->tensors()->Get(output_tensor_index);
            printTensorInfo(output_tensor, model);
        }
    }

    std::cout << "======================" << std::endl;
}



