#include <iostream>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/error_reporter.h>

const char* model_path = "/model/mobilenet_v2.tflite";

std::vector<float> generate_input_data(int input_size) {
    std::vector<float> input_data(input_size, 0.0f);

    for (int i = 0; i < input_size; ++i) {
        input_data[i] = static_cast<float>(i) / input_size;
    }
    return input_data;
}

int main() {
    
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return -1;
    }

    TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    int input_size = input_tensor->bytes / sizeof(float);

    std::vector<float> input_data = generate_input_data(input_size);

    std::copy(input_data.begin(), input_data.end(), interpreter->typed_input_tensor<float>(0));

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke TFLite interpreter" << std::endl;
        return -1;
    }

    TfLiteTensor* output_tensor = interpreter->output_tensor(0);
    float* output_data = output_tensor->data.f;

    for (int i = 0; i < output_tensor->dims->data[1]; ++i) {
        std::cout << "Output " << i << ": " << output_data[i] << std::endl;
    }

    return 0;
}
