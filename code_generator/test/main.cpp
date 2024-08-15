#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "code_generator.h"

using namespace std;
using namespace cv;

uint16_t image_input_x = 0;
uint16_t image_input_y = 0;

int8_t bit8_quantization(float value, float min, float max) {
    float scale = (max - min) / 255.0;
    int quantized_value = round((value - min) / scale);

    if(quantized_value < -128) quantized_value = - 128;
    if(quantized_value > 127) quantized_value = 127;

    return static_cast<int8_t>(quantized_value);
};

int main()
{
    // set image and model path
    const char* img_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/test/person2.png";
    const char* model_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/person_detection_model.tflite";

    // set tensorflow model
    unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if(!model) { cerr << "Failed to load model" << std::endl; return -1;}

    //read input image
    Mat img = imread(img_path, IMREAD_COLOR);
    Mat resize_img;
    resize(img, resize_img, Size(224, 224));

    // (H, W, C) -> (H * W * C) 
    Mat flat_img = resize_img.reshape(1, 1);
    std::cout << flat_img.cols << std::endl;
    uint32_t imageBuffer_size = flat_img.cols;
    uint8_t* imageBuffer = new uint8_t[imageBuffer_size];

    int iterations = 10;  

    for(int i = 0; i < imageBuffer_size; ++i) {
        imageBuffer[i] = flat_img.at<uchar>(i);  
    }
    
    CodeGenerator cg1;
    cg1.setImageInputAnd8bitDataBuffer(
        resize_img.rows, resize_img.cols, resize_img.channels(), 
        imageBuffer, imageBuffer_size
    );

    cg1.parseTFModel(model->GetModel());

    delete[] imageBuffer;

    return 0;
}