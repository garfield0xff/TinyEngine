#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "profiler.h"
#include "codegen.h"


int main() {
    const char* model_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/person_detection_model.tflite";
    const char* save_file_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/model_info.txt";
    const char* img_path = "test_image_dataset/person3.png";
    const char* save_cpp_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/";
    
    // model ownership (not copy)
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    const tflite::Model* tf_model = model->GetModel();

    CodeGenerator c1;
    c1.setCodeGenPath(save_cpp_path);
    c1.parseModel(tf_model);

    // tflite::ops::builtin::BuiltinOpResolver resolver;
    // tflite::InterpreterBuilder builder(*model, resolver);
    // std::unique_ptr<tflite::Interpreter> interpreter;
    // builder(&interpreter);


    // TensorProfiler tp1;
    
    // cv::Mat input_img = cv::imread(img_path);


    // tp1.saveModelInfo(tf_model, save_file_path);
    // tp1.runInference(interpreter.get(), input_img);


    return 0;
}
