#ifndef TENSOR_PROFILER_H
#define TENSOR_PROFILER_H

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <opencv2/opencv.hpp>


using namespace tflite;

class TensorProfiler {
public:
    void saveModelInfo(const tflite::Model* tf_model, const string& output_path );
    void runInference(tflite::Interpreter *interpreter, const cv::Mat& input_img);
    
};

#endif //TENSOR_PROFILER_H