#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "code_generator.h"
#include <fstream>

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
    const char* img_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/test/notPerson1.png";
    const char* model_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/model/person_detection_model.tflite";

    // set tensorflow model
    unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if(!model) { cerr << "Failed to load model" << std::endl; return -1;}

    //read input image
    Mat img = imread(img_path);

    Mat resize_img;
    resize(img, resize_img, Size(224, 224));
    resize_img.convertTo(resize_img, CV_32F, 1.0 / 255.0);

    Mat blob = resize_img.reshape(1, 1);
    

    blob = dnn::blobFromImage(blob);
    // (H, W, C) -> (H * W * C) 
    
    
    // flat_img.convertTo(flat_img, CV_32F, 1.0 / 127.5, -1.0);
    std::cout << blob.size();

    std::cout << blob.size << std::endl;
    uint32_t imageBuffer_size = resize_img.cols * resize_img.rows * resize_img.channels();
    std::cout << imageBuffer_size << std::endl;
    float* imageBuffer = new float[imageBuffer_size];


    int batch_size = blob.size[0];
    std::cout << "Batch size: " << batch_size << std::endl;

    
    for(int i = 0; i < imageBuffer_size; ++i) {
        imageBuffer[i] = resize_img.at<float>(i);  
    }

    // const std::string layer_path = "/Users/gyujinkim/Desktop/Ai/TinyEngine/code_generator/test/";
    // std::ofstream out_file(layer_path + "image.txt");

    // if (out_file.is_open()) {
    //     for (int c = 0; c < resize_img.channels(); ++c) {
    //         out_file << "Channel " << c + 1 << ":" << std::endl;

    //         for (int y = 0; y < resize_img.cols; ++y) {
    //             for (int x = 0; x < resize_img.rows; ++x) {
    //                 int index = (y * resize_img.rows + x) * resize_img.channels() + c;
    //                 out_file << static_cast<int>(imageBuffer[index]) << " ";
    //             }
    //             out_file << std::endl;
    //         }

    //         out_file << std::endl;  
    //     }
    //     out_file.close(); // 파일을 닫습니다.
    // } else {
    //     std::cerr << "Unable to open file for writing!" << std::endl;
    // }

    CodeGenerator cg1;
    cg1.setImageInputAnd8bitDataBuffer(
        resize_img.rows, resize_img.cols, resize_img.channels(), 
        imageBuffer, imageBuffer_size
    );

    cg1.parseTFModel(model->GetModel());

    delete[] imageBuffer;


    // Mat restored_img = Mat(224, 224, CV_32FC3, imageBuffer).clone();
    // restored_img.convertTo(restored_img, CV_8UC3, 255.0);
    // imshow("Restored Image", restored_img);
    // waitKey(0);
    // destroyAllWindows();

    return 0;
}