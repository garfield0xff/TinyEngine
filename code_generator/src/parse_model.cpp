#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

// 모델 파일 경로
const char* model_path = "mobilenet_v2.tflite";

std::unique_ptr<tflite::FlatBufferModel> loadModel(const char* model_path) {
    return tflite::FlatBufferModel::BuildFromFile(model_path);
}

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

int main() {
    std::unique_ptr<tflite::FlatBufferModel> model = loadModel(model_path);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    parseModel(model);
    return 0;
}
