#ifndef CODE_GENERATOR_H
#define CODE_GENERATOR_H

#include <string>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <filesystem>
#include <ostream>

using namespace std;

class CodeGenerator {
protected:
    struct Layer {
        string type;
        vector<int> params;
    };

    stringstream gen_code;
    vector<Layer> layers;
    string codegen_path;

public:
    CodeGenerator();
    ~CodeGenerator();

    void parseModel(const tflite::Model* tf_model);
    void genHeaders();
    void genHeaderFile(set<string> op_headers_set);
    void setCodeGenPath(const string& condegen_path);
    

    void genConv2D(const string& op_func, const tflite::Conv2DOptions* conv_options);
    void genDWConv2D(const string& op_func, const tflite::DepthwiseConv2DOptions* conv_options);
    void genADD(const string& op_func);
    void genPAD(const string& op_func);
    void genMean(const string& op_func);
    void genFullyConnectedLayer(const string& op_func);
    void genLogisticRegression(const string& op_func);
    // void addLayer(const string& layerType, const vector<int>& params);
    // void generateCode(const string& outputPath);
    // void optimizePipeline();
    
    
    // void generateReLU(const Layer& layer);
    // void generateMaxPooling(const Layer& layer);
    // void generateDense(const Layer& layer);
    // void applyLoopUnrolling(string& code);
    
    void genCppModel(const string& output_path, bool is_header);
};

#endif // CODE_GENERATOR_H