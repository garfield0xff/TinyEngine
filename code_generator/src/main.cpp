#include <iostream>
#include <fstream>
#include <vector>
#include "profiler.h"

int main() {
    const char* model_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/mobilenet_v2.tflite";

    // 모델 읽기
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    // 모델의 기본 정보를 출력
    const tflite::Model* tf_model = model->GetModel();
    if (tf_model->subgraphs()->size() == 0) {
        std::cerr << "Model has no subgraphs" << std::endl;
        return -1;
    }

    const tflite::SubGraph* subgraph = tf_model->subgraphs()->Get(0);
    std::cout << "Number of Tensors: " << subgraph->tensors()->size() << std::endl;
    std::cout << "Number of Operators: " << subgraph->operators()->size() << std::endl;

    std::ofstream outfile("/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/model_info.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing" << std::endl;
        return -1;
    }

    TensorProfiler tp1;

    // 각 연산자와 텐서 정보를 출력
    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
        auto op = subgraph->operators()->Get(i);
        tp1.printOperatorInfo(op, tf_model, subgraph);
    }

    return 0;
}
