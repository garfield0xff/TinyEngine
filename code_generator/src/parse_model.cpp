#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>


void parseModel(std::unique_ptr<tflite::FlatBufferModel>& model) {
    const tflite::Model* tf_model = model->GetModel();
    const auto* subgraphs = tf_model->subgraphs();
    for (const auto* subgraph : *subgraphs) {
        for (const auto* op : *subgraph->operators()) {
            auto opcode_index = op->opcode_index();
            auto opcode = tf_model->operator_codes()->Get(opcode_index)->builtin_code();
            // 연산자 종류에 따라 처리
            switch (opcode) {
                case tflite::BuiltinOperator_CONV_2D:
                    std::cout << "Found Conv2D Operator" << std::endl;
                    // Conv2D 연산자 처리 로직
                    break;
                case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
                    std::cout << "Found Depthwise Conv2D Operator" << std::endl;
                    // Depthwise Conv2D 연산자 처리 로직
                    break;
                // 다른 연산자들...
                default:
                    std::cout << "Found other Operator" << std::endl;
                    // 기타 연산자 처리 로직
                    break;
            }
        }
    }
}

